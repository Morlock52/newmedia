// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts-upgradeable/token/ERC721/ERC721Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC721/extensions/ERC721URIStorageUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC721/extensions/ERC721RoyaltyUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/PausableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

/**
 * @title MediaNFTOptimized
 * @dev Gas-optimized NFT contract for media content with advanced features
 * Implements EIP-2981 royalty standard and upgradeable pattern
 */
contract MediaNFTOptimized is 
    Initializable,
    ERC721Upgradeable,
    ERC721URIStorageUpgradeable,
    ERC721RoyaltyUpgradeable,
    AccessControlUpgradeable,
    PausableUpgradeable,
    ReentrancyGuardUpgradeable,
    UUPSUpgradeable 
{
    // Roles
    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");
    bytes32 public constant MODERATOR_ROLE = keccak256("MODERATOR_ROLE");
    
    // State variables packed for gas optimization
    struct MediaData {
        uint128 fileSize;        // File size in bytes
        uint64 creationTime;     // Creation timestamp
        uint32 contentType;      // Enum for content type
        uint32 licenseType;      // License type flags
        bool isEncrypted;        // Whether content is encrypted
        bool isLicensable;       // Can be licensed
        bool isPinned;           // Pinned on IPFS
    }
    
    // Content types enum
    enum ContentType {
        VIDEO,
        AUDIO,
        IMAGE,
        EBOOK,
        DOCUMENT,
        THREED_MODEL,
        VR_EXPERIENCE
    }
    
    // License flags (bit flags for gas efficiency)
    uint32 constant LICENSE_PERSONAL = 1;
    uint32 constant LICENSE_COMMERCIAL = 2;
    uint32 constant LICENSE_RESALE = 4;
    uint32 constant LICENSE_MODIFY = 8;
    uint32 constant LICENSE_DISTRIBUTE = 16;
    
    // Mappings
    mapping(uint256 => MediaData) private _mediaData;
    mapping(uint256 => string) private _ipfsHashes;
    mapping(uint256 => string) private _encryptionKeys;
    mapping(address => uint256[]) private _creatorTokens;
    mapping(string => uint256) private _hashToTokenId;
    
    // Events
    event MediaMinted(
        uint256 indexed tokenId,
        address indexed creator,
        string ipfsHash,
        ContentType contentType
    );
    
    event MediaUpdated(
        uint256 indexed tokenId,
        string newIpfsHash,
        string reason
    );
    
    event LicenseGranted(
        uint256 indexed tokenId,
        address indexed licensee,
        uint32 licenseType,
        uint256 duration
    );
    
    // Custom errors for gas optimization
    error InvalidIPFSHash();
    error ContentAlreadyExists();
    error Unauthorized();
    error InvalidContentType();
    error InvalidLicenseType();
    
    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }
    
    /**
     * @dev Initialize the contract
     */
    function initialize(
        string memory name,
        string memory symbol
    ) public initializer {
        __ERC721_init(name, symbol);
        __ERC721URIStorage_init();
        __ERC721Royalty_init();
        __AccessControl_init();
        __Pausable_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();
        
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
        _grantRole(UPGRADER_ROLE, msg.sender);
    }
    
    /**
     * @dev Mint new media NFT with optimized gas usage
     */
    function mintMedia(
        string calldata ipfsHash,
        ContentType contentType,
        uint128 fileSize,
        uint32 licenseType,
        bool isEncrypted,
        string calldata encryptionKey,
        uint96 royaltyBasisPoints // 10000 = 100%
    ) external whenNotPaused returns (uint256) {
        if (bytes(ipfsHash).length == 0) revert InvalidIPFSHash();
        if (_hashToTokenId[ipfsHash] != 0) revert ContentAlreadyExists();
        if (uint256(contentType) > uint256(ContentType.VR_EXPERIENCE)) revert InvalidContentType();
        
        uint256 tokenId = uint256(keccak256(abi.encodePacked(
            msg.sender,
            block.timestamp,
            ipfsHash
        )));
        
        // Pack data efficiently
        _mediaData[tokenId] = MediaData({
            fileSize: fileSize,
            creationTime: uint64(block.timestamp),
            contentType: uint32(contentType),
            licenseType: licenseType,
            isEncrypted: isEncrypted,
            isLicensable: licenseType > 0,
            isPinned: false
        });
        
        _ipfsHashes[tokenId] = ipfsHash;
        _hashToTokenId[ipfsHash] = tokenId;
        
        if (isEncrypted) {
            _encryptionKeys[tokenId] = encryptionKey;
        }
        
        _creatorTokens[msg.sender].push(tokenId);
        
        _safeMint(msg.sender, tokenId);
        _setTokenURI(tokenId, ipfsHash);
        
        // Set royalty info (EIP-2981)
        if (royaltyBasisPoints > 0) {
            _setTokenRoyalty(tokenId, msg.sender, royaltyBasisPoints);
        }
        
        emit MediaMinted(tokenId, msg.sender, ipfsHash, contentType);
        
        return tokenId;
    }
    
    /**
     * @dev Batch mint for gas efficiency
     */
    function batchMintMedia(
        string[] calldata ipfsHashes,
        ContentType[] calldata contentTypes,
        uint128[] calldata fileSizes,
        uint32 licenseType,
        uint96 royaltyBasisPoints
    ) external whenNotPaused {
        uint256 length = ipfsHashes.length;
        if (length != contentTypes.length || length != fileSizes.length) {
            revert("Array lengths mismatch");
        }
        
        for (uint256 i = 0; i < length;) {
            mintMedia(
                ipfsHashes[i],
                contentTypes[i],
                fileSizes[i],
                licenseType,
                false,
                "",
                royaltyBasisPoints
            );
            
            unchecked { ++i; }
        }
    }
    
    /**
     * @dev Get media data with minimal gas usage
     */
    function getMediaData(uint256 tokenId) external view returns (
        string memory ipfsHash,
        MediaData memory data
    ) {
        require(_exists(tokenId), "Token does not exist");
        return (_ipfsHashes[tokenId], _mediaData[tokenId]);
    }
    
    /**
     * @dev Check if license type is granted
     */
    function hasLicenseType(uint256 tokenId, uint32 licenseFlag) external view returns (bool) {
        return (_mediaData[tokenId].licenseType & licenseFlag) != 0;
    }
    
    /**
     * @dev Update IPFS hash (only creator or moderator)
     */
    function updateIPFSHash(
        uint256 tokenId,
        string calldata newIpfsHash,
        string calldata reason
    ) external {
        require(_exists(tokenId), "Token does not exist");
        require(
            ownerOf(tokenId) == msg.sender || hasRole(MODERATOR_ROLE, msg.sender),
            "Unauthorized"
        );
        
        // Remove old hash mapping
        delete _hashToTokenId[_ipfsHashes[tokenId]];
        
        // Update to new hash
        _ipfsHashes[tokenId] = newIpfsHash;
        _hashToTokenId[newIpfsHash] = tokenId;
        _setTokenURI(tokenId, newIpfsHash);
        
        emit MediaUpdated(tokenId, newIpfsHash, reason);
    }
    
    /**
     * @dev Mark content as pinned on IPFS
     */
    function setPinned(uint256 tokenId, bool pinned) external {
        require(
            ownerOf(tokenId) == msg.sender || hasRole(DEFAULT_ADMIN_ROLE, msg.sender),
            "Unauthorized"
        );
        _mediaData[tokenId].isPinned = pinned;
    }
    
    /**
     * @dev Get all tokens created by an address
     */
    function getCreatorTokens(address creator) external view returns (uint256[] memory) {
        return _creatorTokens[creator];
    }
    
    /**
     * @dev Pause contract (emergency use)
     */
    function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
    }
    
    /**
     * @dev Unpause contract
     */
    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
    
    /**
     * @dev Authorize upgrade (UUPS pattern)
     */
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(UPGRADER_ROLE) {}
    
    // Required overrides
    function _burn(uint256 tokenId) internal override(ERC721Upgradeable, ERC721URIStorageUpgradeable) {
        super._burn(tokenId);
        delete _mediaData[tokenId];
        delete _ipfsHashes[tokenId];
        delete _encryptionKeys[tokenId];
    }
    
    function tokenURI(uint256 tokenId) public view override(ERC721Upgradeable, ERC721URIStorageUpgradeable) returns (string memory) {
        return super.tokenURI(tokenId);
    }
    
    function supportsInterface(bytes4 interfaceId) public view override(ERC721Upgradeable, ERC721RoyaltyUpgradeable, AccessControlUpgradeable) returns (bool) {
        return super.supportsInterface(interfaceId);
    }
    
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 tokenId,
        uint256 batchSize
    ) internal override whenNotPaused {
        super._beforeTokenTransfer(from, to, tokenId, batchSize);
    }
}