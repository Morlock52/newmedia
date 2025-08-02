// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "./MediaNFTOptimized.sol";
import "./CrossChainPaymentProcessor.sol";

/**
 * @title ContentOwnership
 * @dev Manages content ownership, licensing, and access control for Web3 media platform
 * Integrates with MediaNFTOptimized for NFT-based content ownership
 */
contract ContentOwnership is Ownable, ReentrancyGuard, Pausable {
    
    // Core contract references
    MediaNFTOptimized public immutable mediaNFT;
    CrossChainPaymentProcessor public immutable paymentProcessor;
    
    // License types (bit flags for gas efficiency)
    uint32 public constant LICENSE_VIEW = 1;
    uint32 public constant LICENSE_DOWNLOAD = 2;
    uint32 public constant LICENSE_COMMERCIAL = 4;
    uint32 public constant LICENSE_MODIFY = 8;
    uint32 public constant LICENSE_DISTRIBUTE = 16;
    uint32 public constant LICENSE_EXCLUSIVE = 32;
    
    // Content status enum
    enum ContentStatus {
        DRAFT,
        PUBLISHED,
        PAUSED,
        REMOVED
    }
    
    // Content information structure
    struct ContentInfo {
        uint256 tokenId;
        address creator;
        string title;
        string description;
        string[] tags;
        string ipfsHash;
        string thumbnailHash;
        uint256 fileSize;
        string mimeType;
        ContentStatus status;
        uint256 createdAt;
        uint256 updatedAt;
        bool requiresLicense;
        uint256 viewCount;
        uint256 downloadCount;
    }
    
    // License terms structure
    struct LicenseTerms {
        uint256 priceUSD;          // Price in USD (6 decimals)
        uint32 licenseType;        // Bit flags for license permissions
        uint256 duration;          // License duration in seconds (0 = permanent)
        uint256 maxUses;           // Maximum number of uses (0 = unlimited)
        bool isExclusive;          // Exclusive license flag
        bool isTransferable;       // Can license be transferred
        string customTerms;        // Additional terms in IPFS hash
    }
    
    // User license structure
    struct UserLicense {
        uint256 contentId;
        address licensee;
        uint256 grantedAt;
        uint256 expiresAt;
        uint32 licenseType;
        uint256 usesRemaining;
        bool isActive;
        string licenseHash;        // IPFS hash of license agreement
    }
    
    // State mappings
    mapping(uint256 => ContentInfo) public content;
    mapping(uint256 => LicenseTerms) public licenseTerms;
    mapping(uint256 => mapping(address => UserLicense)) public userLicenses;
    mapping(address => uint256[]) public creatorContent;
    mapping(address => uint256[]) public userLicensedContent;
    mapping(string => uint256) public ipfsHashToContentId;
    mapping(string => bool) public usedTitles;
    
    // Counters
    uint256 public nextContentId = 1;
    uint256 public totalContent = 0;
    uint256 public totalLicenses = 0;
    
    // Platform configuration
    uint256 public platformFeePercentage = 250; // 2.5%
    uint256 public maxRoyaltyPercentage = 1000; // 10%
    
    // Events
    event ContentCreated(
        uint256 indexed contentId,
        uint256 indexed tokenId,
        address indexed creator,
        string title,
        string ipfsHash
    );
    
    event ContentUpdated(
        uint256 indexed contentId,
        string field,
        string newValue
    );
    
    event ContentStatusChanged(
        uint256 indexed contentId,
        ContentStatus oldStatus,
        ContentStatus newStatus
    );
    
    event LicenseTermsSet(
        uint256 indexed contentId,
        uint256 priceUSD,
        uint32 licenseType,
        uint256 duration
    );
    
    event LicenseGranted(
        uint256 indexed contentId,
        address indexed licensee,
        uint256 expiresAt,
        uint32 licenseType
    );
    
    event ContentAccessed(
        uint256 indexed contentId,
        address indexed user,
        string accessType
    );
    
    // Custom errors
    error ContentNotFound();
    error UnauthorizedAccess();
    error InvalidLicenseTerms();
    error LicenseExpired();
    error InsufficientPayment();
    error ContentAlreadyExists();
    error InvalidContentStatus();
    
    constructor(
        address _mediaNFT,
        address _paymentProcessor
    ) {
        mediaNFT = MediaNFTOptimized(_mediaNFT);
        paymentProcessor = CrossChainPaymentProcessor(_paymentProcessor);
    }
    
    /**
     * @dev Create new content and mint as NFT
     */
    function createContent(
        string memory title,
        string memory description,
        string[] memory tags,
        string memory ipfsHash,
        string memory thumbnailHash,
        uint256 fileSize,
        string memory mimeType,
        bool requiresLicense,
        LicenseTerms memory _licenseTerms,
        uint96 royaltyBasisPoints
    ) external whenNotPaused nonReentrant returns (uint256) {
        
        // Validate inputs
        require(bytes(title).length > 0 && bytes(title).length <= 200, "Invalid title");
        require(bytes(ipfsHash).length > 0, "Invalid IPFS hash");
        require(fileSize > 0, "Invalid file size");
        require(!usedTitles[title], "Title already used");
        
        if (ipfsHashToContentId[ipfsHash] != 0) revert ContentAlreadyExists();
        
        uint256 contentId = nextContentId++;
        
        // Determine content type based on MIME type
        MediaNFTOptimized.ContentType contentType = _getContentTypeFromMimeType(mimeType);
        
        // Mint NFT
        uint256 tokenId = mediaNFT.mintMedia(
            ipfsHash,
            contentType,
            uint128(fileSize),
            requiresLicense ? _licenseTerms.licenseType : 0,
            false, // Not encrypted for now
            "", // No encryption key
            royaltyBasisPoints
        );
        
        // Create content record
        content[contentId] = ContentInfo({
            tokenId: tokenId,
            creator: msg.sender,
            title: title,
            description: description,
            tags: tags,
            ipfsHash: ipfsHash,
            thumbnailHash: thumbnailHash,
            fileSize: fileSize,
            mimeType: mimeType,
            status: ContentStatus.DRAFT,
            createdAt: block.timestamp,
            updatedAt: block.timestamp,
            requiresLicense: requiresLicense,
            viewCount: 0,
            downloadCount: 0
        });
        
        // Set license terms if required
        if (requiresLicense) {
            _setLicenseTerms(contentId, _licenseTerms);
        }
        
        // Update mappings
        creatorContent[msg.sender].push(contentId);
        ipfsHashToContentId[ipfsHash] = contentId;
        usedTitles[title] = true;
        totalContent++;
        
        emit ContentCreated(contentId, tokenId, msg.sender, title, ipfsHash);
        
        return contentId;
    }
    
    /**
     * @dev Check if user has valid license for content
     */
    function hasValidLicense(
        uint256 contentId,
        address user,
        uint32 requiredLicenseType
    ) external view returns (bool) {
        ContentInfo storage contentInfo = content[contentId];
        if (contentInfo.creator == address(0)) return false;
        
        // Creator always has access
        if (contentInfo.creator == user) return true;
        
        // NFT owner has access
        if (mediaNFT.ownerOf(contentInfo.tokenId) == user) return true;
        
        // Check if content requires license
        if (!contentInfo.requiresLicense) return true;
        
        // Check user license
        UserLicense storage userLicense = userLicenses[contentId][user];
        if (!userLicense.isActive) return false;
        
        // Check expiration
        if (userLicense.expiresAt != 0 && userLicense.expiresAt <= block.timestamp) {
            return false;
        }
        
        // Check license type
        if ((userLicense.licenseType & requiredLicenseType) == 0) return false;
        
        // Check usage limits
        if (userLicense.usesRemaining == 0) return false;
        
        return true;
    }
    
    /**
     * @dev Internal function to set license terms
     */
    function _setLicenseTerms(uint256 contentId, LicenseTerms memory _licenseTerms) internal {
        if (_licenseTerms.priceUSD == 0 && _licenseTerms.licenseType != 0) revert InvalidLicenseTerms();
        
        licenseTerms[contentId] = _licenseTerms;
        
        emit LicenseTermsSet(
            contentId,
            _licenseTerms.priceUSD,
            _licenseTerms.licenseType,
            _licenseTerms.duration
        );
    }
    
    /**
     * @dev Get content type from MIME type
     */
    function _getContentTypeFromMimeType(string memory mimeType) internal pure returns (MediaNFTOptimized.ContentType) {
        bytes32 mimeHash = keccak256(bytes(mimeType));
        
        // Video types
        if (mimeHash == keccak256(bytes("video/mp4")) ||
            mimeHash == keccak256(bytes("video/avi")) ||
            mimeHash == keccak256(bytes("video/mkv")) ||
            mimeHash == keccak256(bytes("video/mov"))) {
            return MediaNFTOptimized.ContentType.VIDEO;
        }
        
        // Audio types
        if (mimeHash == keccak256(bytes("audio/mp3")) ||
            mimeHash == keccak256(bytes("audio/wav")) ||
            mimeHash == keccak256(bytes("audio/aac")) ||
            mimeHash == keccak256(bytes("audio/flac"))) {
            return MediaNFTOptimized.ContentType.AUDIO;
        }
        
        // Image types
        if (mimeHash == keccak256(bytes("image/jpeg")) ||
            mimeHash == keccak256(bytes("image/png")) ||
            mimeHash == keccak256(bytes("image/gif")) ||
            mimeHash == keccak256(bytes("image/webp"))) {
            return MediaNFTOptimized.ContentType.IMAGE;
        }
        
        // Default to document
        return MediaNFTOptimized.ContentType.DOCUMENT;
    }
    
    /**
     * @dev Get content information
     */
    function getContentInfo(uint256 contentId) external view returns (ContentInfo memory) {
        return content[contentId];
    }
    
    /**
     * @dev Get user's licensed content
     */
    function getUserLicensedContent(address user) external view returns (uint256[] memory) {
        return userLicensedContent[user];
    }
    
    /**
     * @dev Get creator's content
     */
    function getCreatorContent(address creator) external view returns (uint256[] memory) {
        return creatorContent[creator];
    }
}