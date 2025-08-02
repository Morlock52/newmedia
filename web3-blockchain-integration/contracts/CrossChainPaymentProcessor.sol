// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@chainlink/contracts/src/v0.8/interfaces/AggregatorV3Interface.sol";
import "@chainlink/contracts/src/v0.8/interfaces/LinkTokenInterface.sol";
import "@chainlink/contracts/src/v0.8/interfaces/VRFV2WrapperInterface.sol";

/**
 * @title CrossChainPaymentProcessor
 * @dev Handles cryptocurrency payments with cross-chain support and price feeds
 * Supports ETH, stablecoins (USDC, USDT, DAI), and wrapped tokens
 */
contract CrossChainPaymentProcessor is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;
    
    // Payment token info
    struct PaymentToken {
        address tokenAddress;
        address priceFeed;      // Chainlink price feed
        uint8 decimals;
        bool isActive;
        bool isStablecoin;
        uint256 minAmount;      // Minimum payment amount
    }
    
    // Subscription plan
    struct SubscriptionPlan {
        uint256 priceUSD;       // Price in USD (6 decimals)
        uint256 duration;       // Duration in seconds
        string name;
        bool isActive;
    }
    
    // User subscription
    struct Subscription {
        uint256 planId;
        uint256 startTime;
        uint256 endTime;
        address paymentToken;
        uint256 amountPaid;
    }
    
    // Payment record
    struct Payment {
        address payer;
        address recipient;
        address token;
        uint256 amount;
        uint256 timestamp;
        string contentId;
        PaymentType paymentType;
    }
    
    enum PaymentType {
        ONE_TIME,
        SUBSCRIPTION,
        TIP,
        LICENSE
    }
    
    // State variables
    mapping(address => PaymentToken) public paymentTokens;
    mapping(uint256 => SubscriptionPlan) public subscriptionPlans;
    mapping(address => Subscription) public userSubscriptions;
    mapping(address => Payment[]) public userPayments;
    mapping(string => uint256) public contentPrices; // Content ID => price in USD
    
    address[] public supportedTokens;
    uint256 public platformFeePercentage = 250; // 2.5%
    uint256 public nextPlanId = 1;
    
    // Events
    event PaymentReceived(
        address indexed payer,
        address indexed recipient,
        address token,
        uint256 amount,
        PaymentType paymentType
    );
    
    event SubscriptionCreated(
        address indexed user,
        uint256 planId,
        uint256 duration
    );
    
    event TokenAdded(
        address indexed token,
        address priceFeed,
        bool isStablecoin
    );
    
    event ContentPriceSet(
        string contentId,
        uint256 priceUSD
    );
    
    // Errors
    error InvalidToken();
    error InsufficientPayment();
    error SubscriptionActive();
    error InvalidPlan();
    error PaymentFailed();
    
    constructor() {}
    
    /**
     * @dev Add a supported payment token
     */
    function addPaymentToken(
        address _token,
        address _priceFeed,
        uint8 _decimals,
        bool _isStablecoin,
        uint256 _minAmount
    ) external onlyOwner {
        paymentTokens[_token] = PaymentToken({
            tokenAddress: _token,
            priceFeed: _priceFeed,
            decimals: _decimals,
            isActive: true,
            isStablecoin: _isStablecoin,
            minAmount: _minAmount
        });
        
        supportedTokens.push(_token);
        
        emit TokenAdded(_token, _priceFeed, _isStablecoin);
    }
    
    /**
     * @dev Create a subscription plan
     */
    function createSubscriptionPlan(
        uint256 _priceUSD,
        uint256 _duration,
        string memory _name
    ) external onlyOwner returns (uint256) {
        uint256 planId = nextPlanId++;
        
        subscriptionPlans[planId] = SubscriptionPlan({
            priceUSD: _priceUSD,
            duration: _duration,
            name: _name,
            isActive: true
        });
        
        return planId;
    }
    
    /**
     * @dev Set content price in USD
     */
    function setContentPrice(
        string memory _contentId,
        uint256 _priceUSD
    ) external onlyOwner {
        contentPrices[_contentId] = _priceUSD;
        emit ContentPriceSet(_contentId, _priceUSD);
    }
    
    /**
     * @dev Pay for content with any supported token
     */
    function payForContent(
        string memory _contentId,
        address _recipient,
        address _paymentToken
    ) external payable nonReentrant {
        PaymentToken memory token = paymentTokens[_paymentToken];
        if (!token.isActive) revert InvalidToken();
        
        uint256 priceUSD = contentPrices[_contentId];
        if (priceUSD == 0) revert("Content not priced");
        
        uint256 requiredAmount = calculateTokenAmount(priceUSD, _paymentToken);
        
        if (_paymentToken == address(0)) {
            // ETH payment
            if (msg.value < requiredAmount) revert InsufficientPayment();
            
            // Send payment to recipient after fee
            uint256 fee = (msg.value * platformFeePercentage) / 10000;
            uint256 recipientAmount = msg.value - fee;
            
            (bool success, ) = _recipient.call{value: recipientAmount}("");
            if (!success) revert PaymentFailed();
            
        } else {
            // ERC20 payment
            IERC20(_paymentToken).safeTransferFrom(
                msg.sender,
                address(this),
                requiredAmount
            );
            
            uint256 fee = (requiredAmount * platformFeePercentage) / 10000;
            uint256 recipientAmount = requiredAmount - fee;
            
            IERC20(_paymentToken).safeTransfer(_recipient, recipientAmount);
        }
        
        // Record payment
        userPayments[msg.sender].push(Payment({
            payer: msg.sender,
            recipient: _recipient,
            token: _paymentToken,
            amount: _paymentToken == address(0) ? msg.value : requiredAmount,
            timestamp: block.timestamp,
            contentId: _contentId,
            paymentType: PaymentType.ONE_TIME
        }));
        
        emit PaymentReceived(
            msg.sender,
            _recipient,
            _paymentToken,
            _paymentToken == address(0) ? msg.value : requiredAmount,
            PaymentType.ONE_TIME
        );
    }
    
    /**
     * @dev Subscribe to a plan
     */
    function subscribe(
        uint256 _planId,
        address _paymentToken
    ) external payable nonReentrant {
        SubscriptionPlan memory plan = subscriptionPlans[_planId];
        if (!plan.isActive) revert InvalidPlan();
        
        // Check if user already has active subscription
        Subscription memory currentSub = userSubscriptions[msg.sender];
        if (currentSub.endTime > block.timestamp) revert SubscriptionActive();
        
        PaymentToken memory token = paymentTokens[_paymentToken];
        if (!token.isActive) revert InvalidToken();
        
        uint256 requiredAmount = calculateTokenAmount(plan.priceUSD, _paymentToken);
        
        if (_paymentToken == address(0)) {
            // ETH payment
            if (msg.value < requiredAmount) revert InsufficientPayment();
            
            // Keep platform fee
            uint256 fee = (msg.value * platformFeePercentage) / 10000;
            
        } else {
            // ERC20 payment
            IERC20(_paymentToken).safeTransferFrom(
                msg.sender,
                address(this),
                requiredAmount
            );
        }
        
        // Create subscription
        userSubscriptions[msg.sender] = Subscription({
            planId: _planId,
            startTime: block.timestamp,
            endTime: block.timestamp + plan.duration,
            paymentToken: _paymentToken,
            amountPaid: _paymentToken == address(0) ? msg.value : requiredAmount
        });
        
        emit SubscriptionCreated(msg.sender, _planId, plan.duration);
    }
    
    /**
     * @dev Calculate token amount needed for USD price
     */
    function calculateTokenAmount(
        uint256 _priceUSD,
        address _token
    ) public view returns (uint256) {
        PaymentToken memory token = paymentTokens[_token];
        
        if (token.isStablecoin) {
            // For stablecoins, assume 1:1 with USD
            return _priceUSD * (10 ** token.decimals) / 1e6;
        }
        
        // Get price from Chainlink oracle
        AggregatorV3Interface priceFeed = AggregatorV3Interface(token.priceFeed);
        (, int256 price, , , ) = priceFeed.latestRoundData();
        
        // Calculate amount needed
        // price has 8 decimals, _priceUSD has 6 decimals
        uint256 tokenAmount = (_priceUSD * 10 ** token.decimals * 1e8) / (uint256(price) * 1e6);
        
        return tokenAmount;
    }
    
    /**
     * @dev Check if user has active subscription
     */
    function hasActiveSubscription(address _user) external view returns (bool) {
        return userSubscriptions[_user].endTime > block.timestamp;
    }
    
    /**
     * @dev Get user payment history
     */
    function getUserPayments(address _user) external view returns (Payment[] memory) {
        return userPayments[_user];
    }
    
    /**
     * @dev Withdraw platform fees
     */
    function withdrawFees(address _token) external onlyOwner {
        if (_token == address(0)) {
            // Withdraw ETH
            uint256 balance = address(this).balance;
            (bool success, ) = owner().call{value: balance}("");
            require(success, "ETH withdrawal failed");
        } else {
            // Withdraw ERC20
            IERC20 token = IERC20(_token);
            uint256 balance = token.balanceOf(address(this));
            token.safeTransfer(owner(), balance);
        }
    }
    
    /**
     * @dev Update platform fee percentage
     */
    function updatePlatformFee(uint256 _newFee) external onlyOwner {
        require(_newFee <= 1000, "Fee too high"); // Max 10%
        platformFeePercentage = _newFee;
    }
    
    /**
     * @dev Emergency pause token
     */
    function pauseToken(address _token) external onlyOwner {
        paymentTokens[_token].isActive = false;
    }
    
    /**
     * @dev Get all supported tokens
     */
    function getSupportedTokens() external view returns (address[] memory) {
        return supportedTokens;
    }
}