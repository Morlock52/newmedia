// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title MediaDAO
 * @dev Decentralized Autonomous Organization for media platform governance
 * Includes voting tokens, proposal system, and governance mechanisms
 */
contract MediaDAO is ERC20, Ownable, ReentrancyGuard {
    using Counters for Counters.Counter;
    
    Counters.Counter private _proposalIdCounter;
    
    // Governance token settings
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 10**18; // 100M tokens
    uint256 public constant MIN_PROPOSAL_THRESHOLD = 1000 * 10**18; // 1K tokens to propose
    uint256 public constant MIN_VOTING_THRESHOLD = 100 * 10**18; // 100 tokens to vote
    uint256 public constant PROPOSAL_DURATION = 7 days;
    uint256 public constant MIN_QUORUM = 5; // 5% of total supply
    
    // Proposal types
    enum ProposalType {
        CONTENT_MODERATION,
        PLATFORM_FEATURE,
        ECONOMIC_PARAMETER,
        GOVERNANCE_CHANGE,
        EMERGENCY_ACTION
    }
    
    // Proposal status
    enum ProposalStatus {
        ACTIVE,
        PASSED,
        FAILED,
        EXECUTED,
        CANCELLED
    }
    
    // Vote choice
    enum VoteChoice {
        AGAINST,
        FOR,
        ABSTAIN
    }
    
    // Proposal structure
    struct Proposal {
        uint256 id;
        address proposer;
        string title;
        string description;
        string[] targets; // Addresses or identifiers affected
        bytes[] calldatas; // Function calls to execute
        uint256 startTime;
        uint256 endTime;
        uint256 forVotes;
        uint256 againstVotes;
        uint256 abstainVotes;
        ProposalType proposalType;
        ProposalStatus status;
        bool executed;
        mapping(address => bool) hasVoted;
        mapping(address => VoteChoice) votes;
        address[] voters;
    }
    
    // Content moderation proposal
    struct ModerationProposal {
        uint256 proposalId;
        string contentHash; // IPFS hash or content identifier
        string contentType;
        address contentCreator;
        string moderationReason;
        bool removeContent;
        bool suspendCreator;
        uint256 suspensionDuration;
    }
    
    // Economic parameter proposal
    struct EconomicProposal {
        uint256 proposalId;
        string parameterName;
        uint256 currentValue;
        uint256 proposedValue;
        string rationale;
    }
    
    // Mappings
    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => ModerationProposal) public moderationProposals;
    mapping(uint256 => EconomicProposal) public economicProposals;
    mapping(address => uint256) public delegatedVotes;
    mapping(address => address) public delegates;
    mapping(address => uint256) public lastRewardClaim;
    
    // Arrays for tracking
    uint256[] public activeProposals;
    uint256[] public executedProposals;
    
    // Economic parameters
    uint256 public platformFeePercentage = 250; // 2.5%
    uint256 public creatorRoyaltyMax = 5000; // 50%
    uint256 public stakingRewardRate = 500; // 5% APY
    uint256 public contentModerationBond = 1000 * 10**18; // 1K tokens
    
    // Events
    event ProposalCreated(
        uint256 indexed proposalId,
        address indexed proposer,
        string title,
        ProposalType proposalType
    );
    
    event VoteCast(
        uint256 indexed proposalId,
        address indexed voter,
        VoteChoice choice,
        uint256 weight
    );
    
    event ProposalExecuted(uint256 indexed proposalId);
    
    event ProposalCancelled(uint256 indexed proposalId);
    
    event DelegateChanged(
        address indexed delegator,
        address indexed fromDelegate,
        address indexed toDelegate
    );
    
    event RewardClaimed(address indexed user, uint256 amount);
    
    event ParameterChanged(string parameter, uint256 oldValue, uint256 newValue);
    
    constructor() ERC20("MediaDAO Token", "MDAO") {
        _mint(msg.sender, INITIAL_SUPPLY);
    }
    
    /**
     * @dev Create a new governance proposal
     * @param title Proposal title
     * @param description Detailed proposal description
     * @param targets Array of target addresses or identifiers
     * @param calldatas Array of function calls to execute
     * @param proposalType Type of proposal
     */
    function createProposal(
        string memory title,
        string memory description,
        string[] memory targets,
        bytes[] memory calldatas,
        ProposalType proposalType
    ) public returns (uint256) {
        require(balanceOf(msg.sender) >= MIN_PROPOSAL_THRESHOLD, "Insufficient tokens to propose");
        require(bytes(title).length > 0, "Title required");
        require(bytes(description).length > 0, "Description required");
        
        uint256 proposalId = _proposalIdCounter.current();
        _proposalIdCounter.increment();
        
        Proposal storage newProposal = proposals[proposalId];
        newProposal.id = proposalId;
        newProposal.proposer = msg.sender;
        newProposal.title = title;
        newProposal.description = description;
        newProposal.targets = targets;
        newProposal.calldatas = calldatas;
        newProposal.startTime = block.timestamp;
        newProposal.endTime = block.timestamp + PROPOSAL_DURATION;
        newProposal.proposalType = proposalType;
        newProposal.status = ProposalStatus.ACTIVE;
        
        activeProposals.push(proposalId);
        
        emit ProposalCreated(proposalId, msg.sender, title, proposalType);
        
        return proposalId;
    }
    
    /**
     * @dev Create content moderation proposal
     * @param contentHash IPFS hash or content identifier
     * @param contentType Type of content
     * @param contentCreator Address of content creator
     * @param moderationReason Reason for moderation
     * @param removeContent Whether to remove content
     * @param suspendCreator Whether to suspend creator
     * @param suspensionDuration Duration of suspension in seconds
     */
    function createModerationProposal(
        string memory contentHash,
        string memory contentType,
        address contentCreator,
        string memory moderationReason,
        bool removeContent,
        bool suspendCreator,
        uint256 suspensionDuration
    ) public returns (uint256) {
        require(balanceOf(msg.sender) >= contentModerationBond, "Insufficient moderation bond");
        
        // Transfer moderation bond
        _transfer(msg.sender, address(this), contentModerationBond);
        
        string memory title = string(abi.encodePacked("Content Moderation: ", contentHash));
        string[] memory targets = new string[](1);
        targets[0] = contentHash;
        bytes[] memory calldatas = new bytes[](1);
        
        uint256 proposalId = createProposal(
            title,
            moderationReason,
            targets,
            calldatas,
            ProposalType.CONTENT_MODERATION
        );
        
        moderationProposals[proposalId] = ModerationProposal({
            proposalId: proposalId,
            contentHash: contentHash,
            contentType: contentType,
            contentCreator: contentCreator,
            moderationReason: moderationReason,
            removeContent: removeContent,
            suspendCreator: suspendCreator,
            suspensionDuration: suspensionDuration
        });
        
        return proposalId;
    }
    
    /**
     * @dev Cast vote on a proposal
     * @param proposalId ID of the proposal
     * @param choice Vote choice (FOR, AGAINST, ABSTAIN)
     */
    function castVote(uint256 proposalId, VoteChoice choice) public {
        require(balanceOf(msg.sender) >= MIN_VOTING_THRESHOLD, "Insufficient voting power");
        
        Proposal storage proposal = proposals[proposalId];
        require(proposal.status == ProposalStatus.ACTIVE, "Proposal not active");
        require(block.timestamp <= proposal.endTime, "Voting period ended");
        require(!proposal.hasVoted[msg.sender], "Already voted");
        
        uint256 votingPower = getVotingPower(msg.sender);
        
        proposal.hasVoted[msg.sender] = true;
        proposal.votes[msg.sender] = choice;
        proposal.voters.push(msg.sender);
        
        if (choice == VoteChoice.FOR) {
            proposal.forVotes += votingPower;
        } else if (choice == VoteChoice.AGAINST) {
            proposal.againstVotes += votingPower;
        } else {
            proposal.abstainVotes += votingPower;
        }
        
        emit VoteCast(proposalId, msg.sender, choice, votingPower);
        
        // Check if proposal should be finalized
        _checkProposalFinalization(proposalId);
    }
    
    /**
     * @dev Execute a passed proposal
     * @param proposalId ID of the proposal to execute
     */
    function executeProposal(uint256 proposalId) public nonReentrant {
        Proposal storage proposal = proposals[proposalId];
        require(proposal.status == ProposalStatus.PASSED, "Proposal not passed");
        require(!proposal.executed, "Already executed");
        require(block.timestamp > proposal.endTime, "Voting still active");
        
        proposal.executed = true;
        proposal.status = ProposalStatus.EXECUTED;
        
        // Execute based on proposal type
        if (proposal.proposalType == ProposalType.CONTENT_MODERATION) {
            _executeModerationProposal(proposalId);
        } else if (proposal.proposalType == ProposalType.ECONOMIC_PARAMETER) {
            _executeEconomicProposal(proposalId);
        }
        
        executedProposals.push(proposalId);
        _removeFromActiveProposals(proposalId);
        
        emit ProposalExecuted(proposalId);
    }
    
    /**
     * @dev Get voting power for an address (includes delegated votes)
     * @param account Address to check
     * @return Voting power
     */
    function getVotingPower(address account) public view returns (uint256) {
        return balanceOf(account) + delegatedVotes[account];
    }
    
    /**
     * @dev Delegate voting power to another address
     * @param delegatee Address to delegate to
     */
    function delegate(address delegatee) public {
        address currentDelegate = delegates[msg.sender];
        require(delegatee != currentDelegate, "Already delegated to this address");
        
        uint256 delegatorBalance = balanceOf(msg.sender);
        
        // Remove votes from current delegate
        if (currentDelegate != address(0)) {
            delegatedVotes[currentDelegate] -= delegatorBalance;
        }
        
        // Add votes to new delegate
        if (delegatee != address(0)) {
            delegatedVotes[delegatee] += delegatorBalance;
        }
        
        delegates[msg.sender] = delegatee;
        
        emit DelegateChanged(msg.sender, currentDelegate, delegatee);
    }
    
    /**
     * @dev Claim staking rewards for governance participation
     */
    function claimRewards() public {
        uint256 userBalance = balanceOf(msg.sender);
        require(userBalance > 0, "No tokens to reward");
        
        uint256 lastClaim = lastRewardClaim[msg.sender];
        if (lastClaim == 0) {
            lastClaim = block.timestamp - 365 days; // Max 1 year retroactive
        }
        
        uint256 timeStaked = block.timestamp - lastClaim;
        uint256 rewardAmount = (userBalance * stakingRewardRate * timeStaked) / (10000 * 365 days);
        
        require(rewardAmount > 0, "No rewards to claim");
        
        lastRewardClaim[msg.sender] = block.timestamp;
        
        // Mint rewards
        _mint(msg.sender, rewardAmount);
        
        emit RewardClaimed(msg.sender, rewardAmount);
    }
    
    /**
     * @dev Check if proposal should be finalized
     * @param proposalId ID of the proposal
     */
    function _checkProposalFinalization(uint256 proposalId) internal {
        Proposal storage proposal = proposals[proposalId];
        
        if (block.timestamp > proposal.endTime) {
            uint256 totalVotes = proposal.forVotes + proposal.againstVotes + proposal.abstainVotes;
            uint256 quorumRequired = (totalSupply() * MIN_QUORUM) / 100;
            
            if (totalVotes >= quorumRequired && proposal.forVotes > proposal.againstVotes) {
                proposal.status = ProposalStatus.PASSED;
            } else {
                proposal.status = ProposalStatus.FAILED;
                _removeFromActiveProposals(proposalId);
            }
        }
    }
    
    /**
     * @dev Execute content moderation proposal
     * @param proposalId ID of the moderation proposal
     */
    function _executeModerationProposal(uint256 proposalId) internal {
        ModerationProposal storage modProposal = moderationProposals[proposalId];
        
        // Implementation would integrate with content management system
        // For now, emit events that can be listened to by the media server
        
        if (modProposal.removeContent) {
            // Signal content removal
            emit ParameterChanged("content_removed", 0, proposalId);
        }
        
        if (modProposal.suspendCreator) {
            // Signal creator suspension
            emit ParameterChanged("creator_suspended", modProposal.suspensionDuration, proposalId);
        }
    }
    
    /**
     * @dev Execute economic parameter proposal
     * @param proposalId ID of the economic proposal
     */
    function _executeEconomicProposal(uint256 proposalId) internal {
        EconomicProposal storage ecoProposal = economicProposals[proposalId];
        
        if (keccak256(bytes(ecoProposal.parameterName)) == keccak256(bytes("platformFeePercentage"))) {
            uint256 oldValue = platformFeePercentage;
            platformFeePercentage = ecoProposal.proposedValue;
            emit ParameterChanged("platformFeePercentage", oldValue, ecoProposal.proposedValue);
        }
        // Add other economic parameters as needed
    }
    
    /**
     * @dev Remove proposal from active proposals array
     * @param proposalId ID of the proposal to remove
     */
    function _removeFromActiveProposals(uint256 proposalId) internal {
        for (uint256 i = 0; i < activeProposals.length; i++) {
            if (activeProposals[i] == proposalId) {
                activeProposals[i] = activeProposals[activeProposals.length - 1];
                activeProposals.pop();
                break;
            }
        }
    }
    
    /**
     * @dev Get all active proposals
     * @return Array of active proposal IDs
     */
    function getActiveProposals() public view returns (uint256[] memory) {
        return activeProposals;
    }
    
    /**
     * @dev Get proposal details
     * @param proposalId ID of the proposal
     * @return Proposal data (without mappings)
     */
    function getProposal(uint256 proposalId) public view returns (
        uint256 id,
        address proposer,
        string memory title,
        string memory description,
        uint256 startTime,
        uint256 endTime,
        uint256 forVotes,
        uint256 againstVotes,
        uint256 abstainVotes,
        ProposalType proposalType,
        ProposalStatus status,
        bool executed
    ) {
        Proposal storage proposal = proposals[proposalId];
        return (
            proposal.id,
            proposal.proposer,
            proposal.title,
            proposal.description,
            proposal.startTime,
            proposal.endTime,
            proposal.forVotes,
            proposal.againstVotes,
            proposal.abstainVotes,
            proposal.proposalType,
            proposal.status,
            proposal.executed
        );
    }
    
    /**
     * @dev Override transfer to update delegated votes
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override {
        super._beforeTokenTransfer(from, to, amount);
        
        // Update delegated votes when tokens are transferred
        if (from != address(0) && delegates[from] != address(0)) {
            delegatedVotes[delegates[from]] -= amount;
        }
        
        if (to != address(0) && delegates[to] != address(0)) {
            delegatedVotes[delegates[to]] += amount;
        }
    }
}