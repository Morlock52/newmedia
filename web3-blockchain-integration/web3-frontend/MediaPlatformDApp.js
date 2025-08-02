/**
 * Web3 Media Platform DApp Frontend
 * React-based decentralized application for Web3 media platform
 * Integrates with smart contracts, IPFS, and existing Jellyfin server
 */

import React, { useState, useEffect, useContext, createContext } from 'react';
import { ethers } from 'ethers';
import { Web3Provider } from '@ethersproject/providers';
import { Contract } from '@ethersproject/contracts';
import { parseEther, formatEther } from '@ethersproject/units';

// Smart Contract ABIs (simplified for example)
const ContentOwnershipABI = [
    "function mintContent(string ipfsHash, string contentType, uint256 fileSize, string title, string description, string[] tags, tuple(uint256 licensePrice, uint256 royaltyPercentage, uint256 licenseDuration, bool commercialUse, bool resaleAllowed, bool modificationAllowed) licenseTerms, bool isLicensable, uint256 maxSupply) returns (uint256)",
    "function purchaseLicense(uint256 tokenId) payable",
    "function hasValidLicense(uint256 tokenId, address user) view returns (bool)",
    "function getContentMetadata(uint256 tokenId) view returns (tuple(address creator, string ipfsHash, string contentType, uint256 creationDate, uint256 fileSize, string title, string description, string[] tags, tuple(uint256 licensePrice, uint256 royaltyPercentage, uint256 licenseDuration, bool commercialUse, bool resaleAllowed, bool modificationAllowed) license, bool isLicensable, uint256 totalSupply, uint256 currentSupply))",
    "function getCreatorContent(address creator) view returns (uint256[])"
];

const MediaDAOABI = [
    "function createProposal(string title, string description, string[] targets, bytes[] calldatas, uint8 proposalType) returns (uint256)",
    "function castVote(uint256 proposalId, uint8 choice)",
    "function getActiveProposals() view returns (uint256[])",
    "function getProposal(uint256 proposalId) view returns (tuple(uint256 id, address proposer, string title, string description, uint256 startTime, uint256 endTime, uint256 forVotes, uint256 againstVotes, uint256 abstainVotes, uint8 proposalType, uint8 status, bool executed))",
    "function balanceOf(address account) view returns (uint256)",
    "function delegate(address delegatee)"
];

const CrossChainMarketplaceABI = [
    "function createListing(tuple(address contractAddress, uint256 tokenId, uint8 blockchain, uint256 chainId, bytes32 assetHash) asset, uint8 listingType, uint256 price, address paymentToken, uint256 duration, uint256 royaltyPercentage, address royaltyRecipient, bool crossChainEnabled, uint8[] supportedChains) returns (uint256)",
    "function purchaseDirect(uint256 listingId, uint8 targetChain) payable",
    "function makeOffer(uint256 listingId, uint256 amount, address paymentToken, uint256 expirationTime, uint8 targetChain) payable returns (uint256)",
    "function getUserListings(address user) view returns (uint256[])",
    "function hasValidLicense(uint256 listingId, address user) view returns (bool)"
];

// Web3 Context
const Web3Context = createContext();

// Web3 Provider Component
export const Web3ContextProvider = ({ children }) => {
    const [provider, setProvider] = useState(null);
    const [signer, setSigner] = useState(null);
    const [account, setAccount] = useState(null);
    const [chainId, setChainId] = useState(null);
    const [contracts, setContracts] = useState({});
    const [isConnected, setIsConnected] = useState(false);
    
    // Contract addresses (would be environment-specific)
    const contractAddresses = {
        ContentOwnership: process.env.REACT_APP_CONTENT_OWNERSHIP_ADDRESS,
        MediaDAO: process.env.REACT_APP_MEDIA_DAO_ADDRESS,
        CrossChainMarketplace: process.env.REACT_APP_MARKETPLACE_ADDRESS
    };
    
    const connectWallet = async () => {
        try {
            if (typeof window.ethereum !== 'undefined') {
                const web3Provider = new Web3Provider(window.ethereum);
                const accounts = await window.ethereum.request({
                    method: 'eth_requestAccounts'
                });
                
                const network = await web3Provider.getNetwork();
                const signer = web3Provider.getSigner();
                
                setProvider(web3Provider);
                setSigner(signer);
                setAccount(accounts[0]);
                setChainId(network.chainId);
                setIsConnected(true);
                
                // Initialize contracts
                const contentOwnership = new Contract(
                    contractAddresses.ContentOwnership,
                    ContentOwnershipABI,
                    signer
                );
                
                const mediaDAO = new Contract(
                    contractAddresses.MediaDAO,
                    MediaDAOABI,
                    signer
                );
                
                const marketplace = new Contract(
                    contractAddresses.CrossChainMarketplace,
                    CrossChainMarketplaceABI,
                    signer
                );
                
                setContracts({
                    contentOwnership,
                    mediaDAO,
                    marketplace
                });
                
                console.log('Wallet connected:', accounts[0]);
                
            } else {
                alert('Please install MetaMask or another Web3 wallet');
            }
        } catch (error) {
            console.error('Error connecting wallet:', error);
        }
    };
    
    const disconnectWallet = () => {
        setProvider(null);
        setSigner(null);
        setAccount(null);
        setChainId(null);
        setContracts({});
        setIsConnected(false);
    };
    
    useEffect(() => {
        // Listen for account changes
        if (window.ethereum) {
            window.ethereum.on('accountsChanged', (accounts) => {
                if (accounts.length === 0) {
                    disconnectWallet();
                } else {
                    setAccount(accounts[0]);
                }
            });
            
            window.ethereum.on('chainChanged', (chainId) => {
                setChainId(parseInt(chainId, 16));
            });
        }
        
        return () => {
            if (window.ethereum) {
                window.ethereum.removeAllListeners('accountsChanged');
                window.ethereum.removeAllListeners('chainChanged');
            }
        };
    }, []);
    
    const value = {
        provider,
        signer,
        account,
        chainId,
        contracts,
        isConnected,
        connectWallet,
        disconnectWallet
    };
    
    return (
        <Web3Context.Provider value={value}>
            {children}
        </Web3Context.Provider>
    );
};

// Hook to use Web3 context
export const useWeb3 = () => {
    const context = useContext(Web3Context);
    if (!context) {
        throw new Error('useWeb3 must be used within a Web3ContextProvider');
    }
    return context;
};

// Wallet Connection Component
export const WalletConnection = () => {
    const { account, isConnected, connectWallet, disconnectWallet, chainId } = useWeb3();
    
    const formatAddress = (address) => {
        return `${address.slice(0, 6)}...${address.slice(-4)}`;
    };
    
    const getNetworkName = (chainId) => {
        const networks = {
            1: 'Ethereum',
            137: 'Polygon',
            56: 'BSC',
            43114: 'Avalanche',
            42161: 'Arbitrum',
            10: 'Optimism'
        };
        return networks[chainId] || 'Unknown';
    };
    
    return (
        <div className="wallet-connection">
            {!isConnected ? (
                <button 
                    onClick={connectWallet}
                    className="connect-wallet-btn"
                >
                    Connect Wallet
                </button>
            ) : (
                <div className="wallet-info">
                    <div className="account-info">
                        <span>{formatAddress(account)}</span>
                        <span className="network">{getNetworkName(chainId)}</span>
                    </div>
                    <button 
                        onClick={disconnectWallet}
                        className="disconnect-btn"
                    >
                        Disconnect
                    </button>
                </div>
            )}
        </div>
    );
};

// Content Upload Component
export const ContentUpload = () => {
    const { contracts, account, isConnected } = useWeb3();
    const [uploadData, setUploadData] = useState({
        title: '',
        description: '',
        contentType: 'video',
        tags: '',
        licensePrice: '',
        royaltyPercentage: '250', // 2.5%
        licenseDuration: '0', // Perpetual
        commercialUse: false,
        resaleAllowed: true,
        modificationAllowed: false,
        isLicensable: true
    });
    const [file, setFile] = useState(null);
    const [ipfsHash, setIpfsHash] = useState('');
    const [uploading, setUploading] = useState(false);
    const [minting, setMinting] = useState(false);
    
    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };
    
    const uploadToIPFS = async () => {
        if (!file) return;
        
        setUploading(true);
        try {
            // This would integrate with your IPFS service
            const formData = new FormData();
            formData.append('file', file);
            formData.append('metadata', JSON.stringify({
                title: uploadData.title,
                description: uploadData.description,
                contentType: uploadData.contentType
            }));
            
            const response = await fetch('/api/ipfs/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            setIpfsHash(result.ipfsHash);
            
        } catch (error) {
            console.error('IPFS upload failed:', error);
            alert('Failed to upload to IPFS');
        } finally {
            setUploading(false);
        }
    };
    
    const mintNFT = async () => {
        if (!ipfsHash || !contracts.contentOwnership) return;
        
        setMinting(true);
        try {
            const tags = uploadData.tags.split(',').map(tag => tag.trim());
            const licenseTerms = {
                licensePrice: parseEther(uploadData.licensePrice || '0'),
                royaltyPercentage: parseInt(uploadData.royaltyPercentage),
                licenseDuration: parseInt(uploadData.licenseDuration),
                commercialUse: uploadData.commercialUse,
                resaleAllowed: uploadData.resaleAllowed,
                modificationAllowed: uploadData.modificationAllowed
            };
            
            const tx = await contracts.contentOwnership.mintContent(
                ipfsHash,
                uploadData.contentType,
                file.size,
                uploadData.title,
                uploadData.description,
                tags,
                licenseTerms,
                uploadData.isLicensable,
                0 // Unlimited supply
            );
            
            await tx.wait();
            alert('Content minted successfully!');
            
            // Reset form
            setUploadData({
                title: '',
                description: '',
                contentType: 'video',
                tags: '',
                licensePrice: '',
                royaltyPercentage: '250',
                licenseDuration: '0',
                commercialUse: false,
                resaleAllowed: true,
                modificationAllowed: false,
                isLicensable: true
            });
            setFile(null);
            setIpfsHash('');
            
        } catch (error) {
            console.error('Minting failed:', error);
            alert('Failed to mint NFT');
        } finally {
            setMinting(false);
        }
    };
    
    if (!isConnected) {
        return (
            <div className="content-upload">
                <p>Please connect your wallet to upload content</p>
            </div>
        );
    }
    
    return (
        <div className="content-upload">
            <h2>Upload Content</h2>
            
            <div className="upload-form">
                <div className="form-group">
                    <label>File:</label>
                    <input 
                        type="file" 
                        onChange={handleFileChange}
                        accept="video/*,audio/*,image/*,.pdf,.epub"
                    />
                </div>
                
                <div className="form-group">
                    <label>Title:</label>
                    <input 
                        type="text"
                        value={uploadData.title}
                        onChange={(e) => setUploadData({...uploadData, title: e.target.value})}
                        placeholder="Content title"
                    />
                </div>
                
                <div className="form-group">
                    <label>Description:</label>
                    <textarea 
                        value={uploadData.description}
                        onChange={(e) => setUploadData({...uploadData, description: e.target.value})}
                        placeholder="Content description"
                    />
                </div>
                
                <div className="form-group">
                    <label>Content Type:</label>
                    <select 
                        value={uploadData.contentType}
                        onChange={(e) => setUploadData({...uploadData, contentType: e.target.value})}
                    >
                        <option value="video">Video</option>
                        <option value="audio">Audio</option>
                        <option value="image">Image</option>
                        <option value="ebook">E-Book</option>
                        <option value="document">Document</option>
                    </select>
                </div>
                
                <div className="form-group">
                    <label>Tags (comma-separated):</label>
                    <input 
                        type="text"
                        value={uploadData.tags}
                        onChange={(e) => setUploadData({...uploadData, tags: e.target.value})}
                        placeholder="tag1, tag2, tag3"
                    />
                </div>
                
                {uploadData.isLicensable && (
                    <>
                        <div className="form-group">
                            <label>License Price (ETH):</label>
                            <input 
                                type="number"
                                step="0.001"
                                value={uploadData.licensePrice}
                                onChange={(e) => setUploadData({...uploadData, licensePrice: e.target.value})}
                                placeholder="0.1"
                            />
                        </div>
                        
                        <div className="form-group">
                            <label>Royalty Percentage (basis points):</label>
                            <input 
                                type="number"
                                value={uploadData.royaltyPercentage}
                                onChange={(e) => setUploadData({...uploadData, royaltyPercentage: e.target.value})}
                                placeholder="250 (2.5%)"
                            />
                        </div>
                        
                        <div className="license-options">
                            <label>
                                <input 
                                    type="checkbox"
                                    checked={uploadData.commercialUse}
                                    onChange={(e) => setUploadData({...uploadData, commercialUse: e.target.checked})}
                                />
                                Commercial Use Allowed
                            </label>
                            
                            <label>
                                <input 
                                    type="checkbox"
                                    checked={uploadData.resaleAllowed}
                                    onChange={(e) => setUploadData({...uploadData, resaleAllowed: e.target.checked})}
                                />
                                Resale Allowed
                            </label>
                            
                            <label>
                                <input 
                                    type="checkbox"
                                    checked={uploadData.modificationAllowed}
                                    onChange={(e) => setUploadData({...uploadData, modificationAllowed: e.target.checked})}
                                />
                                Modification Allowed
                            </label>
                        </div>
                    </>
                )}
                
                <div className="upload-actions">
                    <button 
                        onClick={uploadToIPFS}
                        disabled={!file || uploading}
                        className="upload-btn"
                    >
                        {uploading ? 'Uploading to IPFS...' : 'Upload to IPFS'}
                    </button>
                    
                    {ipfsHash && (
                        <button 
                            onClick={mintNFT}
                            disabled={minting}
                            className="mint-btn"
                        >
                            {minting ? 'Minting NFT...' : 'Mint as NFT'}
                        </button>
                    )}
                </div>
                
                {ipfsHash && (
                    <div className="ipfs-result">
                        <p>IPFS Hash: {ipfsHash}</p>
                    </div>
                )}
            </div>
        </div>
    );
};

// Content Browser Component
export const ContentBrowser = () => {
    const { contracts, account, isConnected } = useWeb3();
    const [content, setContent] = useState([]);
    const [loading, setLoading] = useState(false);
    const [filter, setFilter] = useState('all');
    
    const loadContent = async () => {
        if (!contracts.contentOwnership) return;
        
        setLoading(true);
        try {
            // This would be replaced with a proper indexing service
            const response = await fetch('/api/content/list');
            const contentList = await response.json();
            setContent(contentList);
            
        } catch (error) {
            console.error('Failed to load content:', error);
        } finally {
            setLoading(false);
        }
    };
    
    const purchaseLicense = async (tokenId, price) => {
        if (!contracts.contentOwnership) return;
        
        try {
            const tx = await contracts.contentOwnership.purchaseLicense(tokenId, {
                value: parseEther(price.toString())
            });
            await tx.wait();
            alert('License purchased successfully!');
            
        } catch (error) {
            console.error('License purchase failed:', error);
            alert('Failed to purchase license');
        }
    };
    
    useEffect(() => {
        loadContent();
    }, [contracts.contentOwnership]);
    
    if (!isConnected) {
        return (
            <div className="content-browser">
                <p>Please connect your wallet to browse content</p>
            </div>
        );
    }
    
    return (
        <div className="content-browser">
            <h2>Browse Content</h2>
            
            <div className="browser-controls">
                <select 
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                >
                    <option value="all">All Content</option>
                    <option value="video">Videos</option>
                    <option value="audio">Audio</option>
                    <option value="image">Images</option>
                    <option value="ebook">E-Books</option>
                </select>
                
                <button onClick={loadContent} disabled={loading}>
                    {loading ? 'Loading...' : 'Refresh'}
                </button>
            </div>
            
            <div className="content-grid">
                {content.map((item) => (
                    <div key={item.tokenId} className="content-card">
                        <div className="content-thumbnail">
                            <img 
                                src={`/api/ipfs/thumbnail/${item.ipfsHash}`}
                                alt={item.title}
                                onError={(e) => {
                                    e.target.src = '/placeholder-thumbnail.jpg';
                                }}
                            />
                        </div>
                        
                        <div className="content-info">
                            <h3>{item.title}</h3>
                            <p>{item.description}</p>
                            <div className="content-meta">
                                <span className="type">{item.contentType}</span>
                                <span className="creator">{item.creator}</span>
                            </div>
                            
                            {item.isLicensable && (
                                <div className="license-info">
                                    <span className="price">
                                        {formatEther(item.licensePrice)} ETH
                                    </span>
                                    <button 
                                        onClick={() => purchaseLicense(item.tokenId, formatEther(item.licensePrice))}
                                        className="purchase-btn"
                                    >
                                        Purchase License
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

// DAO Governance Component
export const DAOGovernance = () => {
    const { contracts, account, isConnected } = useWeb3();
    const [proposals, setProposals] = useState([]);
    const [userBalance, setUserBalance] = useState('0');
    const [loading, setLoading] = useState(false);
    const [newProposal, setNewProposal] = useState({
        title: '',
        description: '',
        type: 0 // CONTENT_MODERATION
    });
    
    const loadProposals = async () => {
        if (!contracts.mediaDAO) return;
        
        setLoading(true);
        try {
            const activeProposalIds = await contracts.mediaDAO.getActiveProposals();
            const proposalData = await Promise.all(
                activeProposalIds.map(async (id) => {
                    const proposal = await contracts.mediaDAO.getProposal(id);
                    return { id: id.toString(), ...proposal };
                })
            );
            setProposals(proposalData);
            
            if (account) {
                const balance = await contracts.mediaDAO.balanceOf(account);
                setUserBalance(formatEther(balance));
            }
            
        } catch (error) {
            console.error('Failed to load proposals:', error);
        } finally {
            setLoading(false);
        }
    };
    
    const createProposal = async () => {
        if (!contracts.mediaDAO || !newProposal.title || !newProposal.description) return;
        
        try {
            const tx = await contracts.mediaDAO.createProposal(
                newProposal.title,
                newProposal.description,
                [], // targets
                [], // calldatas
                newProposal.type
            );
            await tx.wait();
            
            alert('Proposal created successfully!');
            setNewProposal({ title: '', description: '', type: 0 });
            loadProposals();
            
        } catch (error) {
            console.error('Failed to create proposal:', error);
            alert('Failed to create proposal');
        }
    };
    
    const castVote = async (proposalId, choice) => {
        if (!contracts.mediaDAO) return;
        
        try {
            const tx = await contracts.mediaDAO.castVote(proposalId, choice);
            await tx.wait();
            alert('Vote cast successfully!');
            loadProposals();
            
        } catch (error) {
            console.error('Failed to cast vote:', error);
            alert('Failed to cast vote');
        }
    };
    
    useEffect(() => {
        loadProposals();
    }, [contracts.mediaDAO, account]);
    
    if (!isConnected) {
        return (
            <div className="dao-governance">
                <p>Please connect your wallet to participate in governance</p>
            </div>
        );
    }
    
    return (
        <div className="dao-governance">
            <h2>DAO Governance</h2>
            
            <div className="user-info">
                <p>Your voting power: {userBalance} MDAO tokens</p>
            </div>
            
            <div className="create-proposal">
                <h3>Create New Proposal</h3>
                <input 
                    type="text"
                    placeholder="Proposal title"
                    value={newProposal.title}
                    onChange={(e) => setNewProposal({...newProposal, title: e.target.value})}
                />
                <textarea 
                    placeholder="Detailed description"
                    value={newProposal.description}
                    onChange={(e) => setNewProposal({...newProposal, description: e.target.value})}
                />
                <select 
                    value={newProposal.type}
                    onChange={(e) => setNewProposal({...newProposal, type: parseInt(e.target.value)})}
                >
                    <option value={0}>Content Moderation</option>
                    <option value={1}>Platform Feature</option>
                    <option value={2}>Economic Parameter</option>
                    <option value={3}>Governance Change</option>
                    <option value={4}>Emergency Action</option>
                </select>
                <button onClick={createProposal}>Create Proposal</button>
            </div>
            
            <div className="proposals-list">
                <h3>Active Proposals</h3>
                {loading ? (
                    <p>Loading proposals...</p>
                ) : (
                    proposals.map((proposal) => (
                        <div key={proposal.id} className="proposal-card">
                            <h4>{proposal.title}</h4>
                            <p>{proposal.description}</p>
                            
                            <div className="voting-stats">
                                <span className="for-votes">
                                    For: {formatEther(proposal.forVotes)}
                                </span>
                                <span className="against-votes">
                                    Against: {formatEther(proposal.againstVotes)}
                                </span>
                                <span className="abstain-votes">
                                    Abstain: {formatEther(proposal.abstainVotes)}
                                </span>
                            </div>
                            
                            <div className="voting-actions">
                                <button 
                                    onClick={() => castVote(proposal.id, 1)}
                                    className="vote-for"
                                >
                                    Vote For
                                </button>
                                <button 
                                    onClick={() => castVote(proposal.id, 0)}
                                    className="vote-against"
                                >
                                    Vote Against
                                </button>
                                <button 
                                    onClick={() => castVote(proposal.id, 2)}
                                    className="vote-abstain"
                                >
                                    Abstain
                                </button>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

// Main DApp Component
export const MediaPlatformDApp = () => {
    const [activeTab, setActiveTab] = useState('browse');
    
    return (
        <Web3ContextProvider>
            <div className="media-platform-dapp">
                <header className="dapp-header">
                    <h1>Decentralized Media Platform</h1>
                    <WalletConnection />
                </header>
                
                <nav className="dapp-navigation">
                    <button 
                        className={activeTab === 'browse' ? 'active' : ''}
                        onClick={() => setActiveTab('browse')}
                    >
                        Browse Content
                    </button>
                    <button 
                        className={activeTab === 'upload' ? 'active' : ''}
                        onClick={() => setActiveTab('upload')}
                    >
                        Upload Content
                    </button>
                    <button 
                        className={activeTab === 'governance' ? 'active' : ''}
                        onClick={() => setActiveTab('governance')}
                    >
                        DAO Governance
                    </button>
                </nav>
                
                <main className="dapp-content">
                    {activeTab === 'browse' && <ContentBrowser />}
                    {activeTab === 'upload' && <ContentUpload />}
                    {activeTab === 'governance' && <DAOGovernance />}
                </main>
            </div>
        </Web3ContextProvider>
    );
};

export default MediaPlatformDApp;