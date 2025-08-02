// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/IERC721.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title CrossChainMarketplace
 * @dev Decentralized marketplace for cross-chain media NFT trading and licensing
 * Supports multiple blockchains, payment tokens, and automated royalty distribution
 */
contract CrossChainMarketplace is ReentrancyGuard, Ownable {
    using Counters for Counters.Counter;
    
    Counters.Counter private _listingIdCounter;
    Counters.Counter private _offerIdCounter;
    
    // Supported blockchain identifiers
    enum Blockchain {
        ETHEREUM,
        POLYGON,
        BSC,
        AVALANCHE,
        ARBITRUM,
        OPTIMISM
    }
    
    // Listing types
    enum ListingType {
        FIXED_PRICE,
        AUCTION,
        LICENSE_ONLY,
        RENTAL
    }
    
    // Listing status
    enum ListingStatus {
        ACTIVE,
        SOLD,
        CANCELLED,
        EXPIRED
    }
    
    // Payment token structure
    struct PaymentToken {
        address tokenAddress;
        string symbol;
        uint8 decimals;
        bool active;
        uint256 chainId;
    }
    
    // Cross-chain asset structure
    struct CrossChainAsset {
        address contractAddress;
        uint256 tokenId;
        Blockchain blockchain;
        uint256 chainId;
        bytes32 assetHash; // Unique identifier across chains
    }
    
    // Marketplace listing
    struct Listing {
        uint256 listingId;
        address seller;
        CrossChainAsset asset;
        ListingType listingType;
        uint256 price;
        address paymentToken;
        uint256 startTime;
        uint256 endTime;
        uint256 royaltyPercentage;
        address royaltyRecipient;
        ListingStatus status;
        bool crossChainEnabled;
        Blockchain[] supportedChains;
    }
    
    // Offer structure for auctions and negotiations
    struct Offer {
        uint256 offerId;
        uint256 listingId;
        address bidder;
        uint256 amount;
        address paymentToken;
        uint256 timestamp;
        uint256 expirationTime;
        bool active;
        Blockchain targetChain;
    }
    
    // License terms for content licensing
    struct LicenseTerms {
        uint256 duration; // Duration in seconds, 0 = perpetual
        bool commercialUse;
        bool resaleAllowed;
        bool modificationAllowed;
        uint256 maxCopies; // 0 = unlimited
        string[] restrictedTerritories;
    }
    
    // License purchase record
    struct LicensePurchase {
        uint256 listingId;
        address licensee;
        uint256 purchaseTime;
        uint256 expirationTime;
        uint256 pricePaid;
        LicenseTerms terms;
        bool active;
    }
    
    // Mappings
    mapping(uint256 => Listing) public listings;
    mapping(uint256 => Offer) public offers;
    mapping(uint256 => LicenseTerms) public listingLicenseTerms;
    mapping(uint256 => mapping(address => LicensePurchase)) public licensePurchases;
    mapping(address => bool) public authorizedPaymentTokens;
    mapping(uint256 => PaymentToken) public paymentTokens;
    mapping(bytes32 => bool) public verifiedCrossChainAssets;
    mapping(address => uint256[]) public userListings;
    mapping(address => uint256[]) public userOffers;
    mapping(Blockchain => address) public chainBridgeContracts;
    
    // Platform settings
    uint256 public platformFeePercentage = 250; // 2.5%
    uint256 public maxRoyaltyPercentage = 2500; // 25%
    address public feeRecipient;
    
    // Events
    event ListingCreated(
        uint256 indexed listingId,
        address indexed seller,
        address indexed contractAddress,
        uint256 tokenId,
        uint256 price,
        ListingType listingType
    );
    
    event OfferMade(
        uint256 indexed offerId,
        uint256 indexed listingId,
        address indexed bidder,
        uint256 amount
    );
    
    event SaleCompleted(
        uint256 indexed listingId,
        address indexed buyer,
        address indexed seller,
        uint256 price,
        address paymentToken
    );
    
    event LicensePurchased(
        uint256 indexed listingId,
        address indexed licensee,
        uint256 price,
        uint256 duration
    );
    
    event CrossChainTransferInitiated(
        uint256 indexed listingId,
        Blockchain fromChain,
        Blockchain toChain,
        bytes32 transferId
    );
    
    event RoyaltyPaid(
        uint256 indexed listingId,
        address indexed recipient,
        uint256 amount
    );
    
    constructor(address _feeRecipient) {
        feeRecipient = _feeRecipient;
        _setupDefaultPaymentTokens();
    }
    
    /**
     * @dev Create a new listing for NFT sale or licensing
     * @param asset Cross-chain asset information
     * @param listingType Type of listing (fixed price, auction, etc.)
     * @param price Listing price
     * @param paymentToken Address of payment token (0x0 for native token)
     * @param duration Duration of listing in seconds
     * @param royaltyPercentage Royalty percentage for creator
     * @param royaltyRecipient Address to receive royalties
     * @param crossChainEnabled Whether listing supports cross-chain purchases
     * @param supportedChains Array of supported blockchains
     */
    function createListing(
        CrossChainAsset memory asset,
        ListingType listingType,
        uint256 price,
        address paymentToken,
        uint256 duration,
        uint256 royaltyPercentage,
        address royaltyRecipient,
        bool crossChainEnabled,
        Blockchain[] memory supportedChains
    ) public returns (uint256) {
        require(price > 0, "Price must be greater than 0");
        require(royaltyPercentage <= maxRoyaltyPercentage, "Royalty too high");
        require(royaltyRecipient != address(0), "Invalid royalty recipient");
        
        // Verify NFT ownership (for same-chain assets)
        if (asset.chainId == block.chainid) {
            IERC721 nftContract = IERC721(asset.contractAddress);
            require(nftContract.ownerOf(asset.tokenId) == msg.sender, "Not NFT owner");
        }
        
        uint256 listingId = _listingIdCounter.current();
        _listingIdCounter.increment();
        
        listings[listingId] = Listing({
            listingId: listingId,
            seller: msg.sender,
            asset: asset,
            listingType: listingType,
            price: price,
            paymentToken: paymentToken,
            startTime: block.timestamp,
            endTime: duration > 0 ? block.timestamp + duration : 0,
            royaltyPercentage: royaltyPercentage,
            royaltyRecipient: royaltyRecipient,
            status: ListingStatus.ACTIVE,
            crossChainEnabled: crossChainEnabled,
            supportedChains: supportedChains
        });
        
        userListings[msg.sender].push(listingId);
        
        emit ListingCreated(
            listingId,
            msg.sender,
            asset.contractAddress,
            asset.tokenId,
            price,
            listingType
        );
        
        return listingId;
    }
    
    /**
     * @dev Create license-only listing with specific terms
     * @param asset Cross-chain asset information
     * @param price License price
     * @param paymentToken Payment token address
     * @param licenseTerms Licensing terms and restrictions
     * @param royaltyPercentage Royalty percentage
     * @param royaltyRecipient Royalty recipient address
     */
    function createLicenseListing(
        CrossChainAsset memory asset,
        uint256 price,
        address paymentToken,
        LicenseTerms memory licenseTerms,
        uint256 royaltyPercentage,
        address royaltyRecipient
    ) public returns (uint256) {
        Blockchain[] memory supportedChains = new Blockchain[](1);
        supportedChains[0] = asset.blockchain;
        
        uint256 listingId = createListing(
            asset,
            ListingType.LICENSE_ONLY,
            price,
            paymentToken,
            0, // No expiration for license listings
            royaltyPercentage,
            royaltyRecipient,
            true, // Enable cross-chain by default
            supportedChains
        );
        
        listingLicenseTerms[listingId] = licenseTerms;
        
        return listingId;
    }
    
    /**
     * @dev Purchase NFT or license directly at fixed price
     * @param listingId ID of the listing to purchase
     * @param targetChain Target blockchain for cross-chain purchase
     */
    function purchaseDirect(uint256 listingId, Blockchain targetChain) public payable nonReentrant {
        Listing storage listing = listings[listingId];
        require(listing.status == ListingStatus.ACTIVE, "Listing not active");
        require(listing.listingType == ListingType.FIXED_PRICE || 
                listing.listingType == ListingType.LICENSE_ONLY, "Not fixed price listing");
        require(listing.endTime == 0 || block.timestamp <= listing.endTime, "Listing expired");
        
        uint256 totalPrice = listing.price;
        
        // Handle payment
        if (listing.paymentToken == address(0)) {
            require(msg.value >= totalPrice, "Insufficient payment");
        } else {
            IERC20 token = IERC20(listing.paymentToken);
            require(token.transferFrom(msg.sender, address(this), totalPrice), "Payment failed");
        }
        
        // Process different listing types
        if (listing.listingType == ListingType.LICENSE_ONLY) {
            _processLicensePurchase(listingId, msg.sender, totalPrice);
        } else {
            _processSale(listingId, msg.sender, totalPrice, targetChain);
        }
    }
    
    /**
     * @dev Make an offer on an auction or negotiate price
     * @param listingId ID of the listing
     * @param amount Offer amount
     * @param paymentToken Payment token address
     * @param expirationTime When offer expires
     * @param targetChain Target blockchain for delivery
     */
    function makeOffer(
        uint256 listingId,
        uint256 amount,
        address paymentToken,
        uint256 expirationTime,
        Blockchain targetChain
    ) public payable returns (uint256) {
        Listing storage listing = listings[listingId];
        require(listing.status == ListingStatus.ACTIVE, "Listing not active");
        require(amount > 0, "Invalid offer amount");
        require(expirationTime > block.timestamp, "Invalid expiration time");
        
        uint256 offerId = _offerIdCounter.current();
        _offerIdCounter.increment();
        
        // Lock payment for offer
        if (paymentToken == address(0)) {
            require(msg.value >= amount, "Insufficient payment for offer");
        } else {
            IERC20 token = IERC20(paymentToken);
            require(token.transferFrom(msg.sender, address(this), amount), "Payment lock failed");
        }
        
        offers[offerId] = Offer({
            offerId: offerId,
            listingId: listingId,
            bidder: msg.sender,
            amount: amount,
            paymentToken: paymentToken,
            timestamp: block.timestamp,
            expirationTime: expirationTime,
            active: true,
            targetChain: targetChain
        });
        
        userOffers[msg.sender].push(offerId);
        
        emit OfferMade(offerId, listingId, msg.sender, amount);
        
        return offerId;
    }
    
    /**
     * @dev Accept an offer (seller only)
     * @param offerId ID of the offer to accept
     */
    function acceptOffer(uint256 offerId) public nonReentrant {
        Offer storage offer = offers[offerId];
        require(offer.active, "Offer not active");
        require(block.timestamp <= offer.expirationTime, "Offer expired");
        
        Listing storage listing = listings[offer.listingId];
        require(listing.seller == msg.sender, "Not the seller");
        require(listing.status == ListingStatus.ACTIVE, "Listing not active");
        
        offer.active = false;
        
        if (listing.listingType == ListingType.LICENSE_ONLY) {
            _processLicensePurchase(offer.listingId, offer.bidder, offer.amount);
        } else {
            _processSale(offer.listingId, offer.bidder, offer.amount, offer.targetChain);
        }
    }
    
    /**
     * @dev Process NFT sale with cross-chain support
     * @param listingId ID of the listing
     * @param buyer Buyer address
     * @param totalPrice Total sale price
     * @param targetChain Target blockchain for delivery
     */
    function _processSale(
        uint256 listingId,
        address buyer,
        uint256 totalPrice,
        Blockchain targetChain
    ) internal {
        Listing storage listing = listings[listingId];
        
        // Calculate fees and royalties
        uint256 platformFee = (totalPrice * platformFeePercentage) / 10000;
        uint256 royalty = (totalPrice * listing.royaltyPercentage) / 10000;
        uint256 sellerAmount = totalPrice - platformFee - royalty;
        
        // Distribute payments
        _distributePayments(
            listing.paymentToken,
            listing.seller,
            listing.royaltyRecipient,
            sellerAmount,
            royalty,
            platformFee
        );
        
        // Handle cross-chain transfer if needed
        if (targetChain != listing.asset.blockchain && listing.crossChainEnabled) {
            _initiateCrossChainTransfer(listingId, buyer, targetChain);
        } else {
            // Same-chain transfer
            IERC721 nftContract = IERC721(listing.asset.contractAddress);
            nftContract.safeTransferFrom(listing.seller, buyer, listing.asset.tokenId);
        }
        
        listing.status = ListingStatus.SOLD;
        
        emit SaleCompleted(listingId, buyer, listing.seller, totalPrice, listing.paymentToken);
    }
    
    /**
     * @dev Process license purchase
     * @param listingId ID of the listing
     * @param licensee Licensee address
     * @param price License price paid
     */
    function _processLicensePurchase(
        uint256 listingId,
        address licensee,
        uint256 price
    ) internal {
        Listing storage listing = listings[listingId];
        LicenseTerms storage terms = listingLicenseTerms[listingId];
        
        // Calculate fees and royalties
        uint256 platformFee = (price * platformFeePercentage) / 10000;
        uint256 royalty = (price * listing.royaltyPercentage) / 10000;
        uint256 sellerAmount = price - platformFee - royalty;
        
        // Distribute payments
        _distributePayments(
            listing.paymentToken,
            listing.seller,
            listing.royaltyRecipient,
            sellerAmount,
            royalty,
            platformFee
        );
        
        // Create license record
        uint256 expirationTime = terms.duration > 0 
            ? block.timestamp + terms.duration 
            : 0; // Perpetual license
            
        licensePurchases[listingId][licensee] = LicensePurchase({
            listingId: listingId,
            licensee: licensee,
            purchaseTime: block.timestamp,
            expirationTime: expirationTime,
            pricePaid: price,
            terms: terms,
            active: true
        });
        
        emit LicensePurchased(listingId, licensee, price, terms.duration);
    }
    
    /**
     * @dev Distribute payments to seller, royalty recipient, and platform
     */
    function _distributePayments(
        address paymentToken,
        address seller,
        address royaltyRecipient,
        uint256 sellerAmount,
        uint256 royalty,
        uint256 platformFee
    ) internal {
        if (paymentToken == address(0)) {
            // Native token payments
            if (sellerAmount > 0) {
                payable(seller).transfer(sellerAmount);
            }
            if (royalty > 0) {
                payable(royaltyRecipient).transfer(royalty);
            }
            if (platformFee > 0) {
                payable(feeRecipient).transfer(platformFee);
            }
        } else {
            // ERC20 token payments
            IERC20 token = IERC20(paymentToken);
            if (sellerAmount > 0) {
                token.transfer(seller, sellerAmount);
            }
            if (royalty > 0) {
                token.transfer(royaltyRecipient, royalty);
            }
            if (platformFee > 0) {
                token.transfer(feeRecipient, platformFee);
            }
        }
    }
    
    /**
     * @dev Initiate cross-chain NFT transfer
     * @param listingId ID of the listing
     * @param buyer Buyer address
     * @param targetChain Target blockchain
     */
    function _initiateCrossChainTransfer(
        uint256 listingId,
        address buyer,
        Blockchain targetChain
    ) internal {
        Listing storage listing = listings[listingId];
        
        // Generate unique transfer ID
        bytes32 transferId = keccak256(
            abi.encodePacked(listingId, buyer, targetChain, block.timestamp)
        );
        
        // This would integrate with cross-chain bridge contracts
        // For now, emit event for external bridge to process
        emit CrossChainTransferInitiated(
            listingId,
            listing.asset.blockchain,
            targetChain,
            transferId
        );
    }
    
    /**
     * @dev Setup default payment tokens
     */
    function _setupDefaultPaymentTokens() internal {
        // Add common stablecoins and tokens
        // This would be configured based on deployment chain
    }
    
    /**
     * @dev Check if user has valid license for content
     * @param listingId ID of the content listing
     * @param user User address to check
     * @return bool Whether user has valid license
     */
    function hasValidLicense(uint256 listingId, address user) public view returns (bool) {
        LicensePurchase storage license = licensePurchases[listingId][user];
        if (!license.active) return false;
        
        // Check expiration
        if (license.expirationTime == 0) return true; // Perpetual license
        return license.expirationTime > block.timestamp;
    }
    
    /**
     * @dev Get user's listings
     * @param user User address
     * @return Array of listing IDs
     */
    function getUserListings(address user) public view returns (uint256[] memory) {
        return userListings[user];
    }
    
    /**
     * @dev Get user's offers
     * @param user User address
     * @return Array of offer IDs
     */
    function getUserOffers(address user) public view returns (uint256[] memory) {
        return userOffers[user];
    }
    
    /**
     * @dev Cancel listing (seller only)
     * @param listingId ID of listing to cancel
     */
    function cancelListing(uint256 listingId) public {
        Listing storage listing = listings[listingId];
        require(listing.seller == msg.sender, "Not the seller");
        require(listing.status == ListingStatus.ACTIVE, "Listing not active");
        
        listing.status = ListingStatus.CANCELLED;
    }
    
    /**
     * @dev Update platform fee (owner only)
     * @param newFeePercentage New fee percentage in basis points
     */
    function updatePlatformFee(uint256 newFeePercentage) public onlyOwner {
        require(newFeePercentage <= 1000, "Fee too high"); // Max 10%
        platformFeePercentage = newFeePercentage;
    }
    
    /**
     * @dev Add supported payment token (owner only)
     * @param tokenAddress Token contract address
     * @param symbol Token symbol
     * @param decimals Token decimals
     */
    function addPaymentToken(
        address tokenAddress,
        string memory symbol,
        uint8 decimals
    ) public onlyOwner {
        authorizedPaymentTokens[tokenAddress] = true;
        // Additional token setup logic
    }
}