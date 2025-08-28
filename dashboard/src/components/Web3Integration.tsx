import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Web3Wallet {
  address: string;
  balance: number;
  network: string;
  connected: boolean;
  provider: 'MetaMask' | 'WalletConnect' | 'Coinbase' | 'Rainbow';
}

interface NFTAsset {
  id: string;
  name: string;
  description: string;
  image: string;
  collection: string;
  owner: string;
  price: number;
  currency: 'ETH' | 'MATIC' | 'SOL';
  blockchain: string;
  metadata: Record<string, any>;
}

interface DeFiPosition {
  id: string;
  protocol: string;
  type: 'lending' | 'staking' | 'liquidity' | 'farming';
  asset: string;
  amount: number;
  apy: number;
  value: number;
  rewards: number;
}

interface BlockchainTransaction {
  hash: string;
  type: 'send' | 'receive' | 'swap' | 'mint' | 'stake';
  amount: number;
  currency: string;
  from: string;
  to: string;
  timestamp: Date;
  status: 'pending' | 'confirmed' | 'failed';
  gasUsed: number;
  gasFee: number;
}

const Web3Integration: React.FC = () => {
  const [wallet, setWallet] = useState<Web3Wallet | null>(null);
  const [nftAssets, setNftAssets] = useState<NFTAsset[]>([]);
  const [defiPositions, setDeFiPositions] = useState<DeFiPosition[]>([]);
  const [transactions, setTransactions] = useState<BlockchainTransaction[]>([]);
  const [selectedNFT, setSelectedNFT] = useState<NFTAsset | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [currentNetwork, setCurrentNetwork] = useState('ethereum');
  const [portfolioValue, setPortfolioValue] = useState(0);
  const [marketData, setMarketData] = useState({
    eth: { price: 2500, change24h: 3.2 },
    btc: { price: 45000, change24h: -1.8 },
    matic: { price: 0.85, change24h: 5.6 }
  });
  const [web3Features, setWeb3Features] = useState({
    nftGallery: true,
    defiIntegration: true,
    daoVoting: false,
    crossChain: true,
    metaverse: false
  });
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const web3Ref = useRef<any>(null);

  useEffect(() => {
    initializeWeb3();
    loadMockData();
    startBlockchainVisualization();
  }, []);

  const initializeWeb3 = async () => {
    // Check if MetaMask or other Web3 provider is available
    if (typeof window !== 'undefined' && (window as any).ethereum) {
      web3Ref.current = (window as any).ethereum;
      
      // Listen for account changes
      web3Ref.current.on('accountsChanged', handleAccountsChanged);
      web3Ref.current.on('chainChanged', handleChainChanged);
      
      // Check if already connected
      try {
        const accounts = await web3Ref.current.request({ method: 'eth_accounts' });
        if (accounts.length > 0) {
          await connectWallet();
        }
      } catch (error) {
        console.log('Not connected to Web3 wallet');
      }
    }
  };

  const connectWallet = async (providerType: string = 'MetaMask') => {
    setIsConnecting(true);
    
    try {
      if (web3Ref.current) {
        const accounts = await web3Ref.current.request({ method: 'eth_requestAccounts' });
        const chainId = await web3Ref.current.request({ method: 'eth_chainId' });
        const balance = await web3Ref.current.request({
          method: 'eth_getBalance',
          params: [accounts[0], 'latest']
        });
        
        const walletData: Web3Wallet = {
          address: accounts[0],
          balance: parseInt(balance, 16) / Math.pow(10, 18), // Convert Wei to ETH
          network: getNetworkName(chainId),
          connected: true,
          provider: providerType as any
        };
        
        setWallet(walletData);
        loadUserAssets(accounts[0]);
        
      } else {
        // Simulate wallet connection for demo
        const mockWallet: Web3Wallet = {
          address: '0x742d35Cc7Da82a2C4C1C24781C3C90C4AEa3b1C5',
          balance: 2.5,
          network: 'Ethereum',
          connected: true,
          provider: 'MetaMask'
        };
        setWallet(mockWallet);
        loadUserAssets(mockWallet.address);
      }
    } catch (error) {
      console.error('Failed to connect wallet:', error);
    } finally {
      setIsConnecting(false);
    }
  };

  const disconnectWallet = () => {
    setWallet(null);
    setNftAssets([]);
    setDeFiPositions([]);
    setTransactions([]);
    setPortfolioValue(0);
  };

  const handleAccountsChanged = (accounts: string[]) => {
    if (accounts.length === 0) {
      disconnectWallet();
    } else if (wallet && accounts[0] !== wallet.address) {
      connectWallet();
    }
  };

  const handleChainChanged = (chainId: string) => {
    const networkName = getNetworkName(chainId);
    setCurrentNetwork(networkName.toLowerCase());
    if (wallet) {
      setWallet({ ...wallet, network: networkName });
    }
  };

  const getNetworkName = (chainId: string) => {
    switch (chainId) {
      case '0x1': return 'Ethereum';
      case '0x89': return 'Polygon';
      case '0xa86a': return 'Avalanche';
      case '0x38': return 'BSC';
      default: return 'Unknown';
    }
  };

  const loadMockData = () => {
    // Mock NFT assets
    const mockNFTs: NFTAsset[] = [
      {
        id: '1',
        name: 'Cyber Nexus #2847',
        description: 'A rare cyberpunk-themed NFT from the Cyber Nexus collection',
        image: 'https://via.placeholder.com/300x300/0a0a0a/00ffff?text=CYBER+NFT',
        collection: 'Cyber Nexus',
        owner: '0x742d35Cc7Da82a2C4C1C24781C3C90C4AEa3b1C5',
        price: 1.5,
        currency: 'ETH',
        blockchain: 'Ethereum',
        metadata: {
          rarity: 'Legendary',
          attributes: [
            { trait_type: 'Background', value: 'Neon City' },
            { trait_type: 'Eyes', value: 'Cyber Blue' },
            { trait_type: 'Accessories', value: 'Neural Implant' }
          ]
        }
      },
      {
        id: '2',
        name: 'Matrix Avatar #9912',
        description: 'Digital identity avatar for the metaverse',
        image: 'https://via.placeholder.com/300x300/1a1a2e/ff00ff?text=MATRIX+AVT',
        collection: 'Matrix Avatars',
        owner: '0x742d35Cc7Da82a2C4C1C24781C3C90C4AEa3b1C5',
        price: 0.8,
        currency: 'ETH',
        blockchain: 'Ethereum',
        metadata: {
          rarity: 'Epic',
          attributes: [
            { trait_type: 'Type', value: 'Hacker' },
            { trait_type: 'Skill', value: 'Data Mining' },
            { trait_type: 'Level', value: '95' }
          ]
        }
      },
      {
        id: '3',
        name: 'Hologram Cube #555',
        description: 'Interactive 3D holographic art piece',
        image: 'https://via.placeholder.com/300x300/16213e/ffff00?text=HOLO+CUBE',
        collection: 'Holographic Art',
        owner: '0x742d35Cc7Da82a2C4C1C24781C3C90C4AEa3b1C5',
        price: 2.2,
        currency: 'ETH',
        blockchain: 'Ethereum',
        metadata: {
          rarity: 'Rare',
          attributes: [
            { trait_type: 'Dimension', value: '4D' },
            { trait_type: 'Animation', value: 'Infinite Loop' },
            { trait_type: 'Interactivity', value: 'High' }
          ]
        }
      }
    ];
    
    // Mock DeFi positions
    const mockDeFi: DeFiPosition[] = [
      {
        id: '1',
        protocol: 'Compound',
        type: 'lending',
        asset: 'ETH',
        amount: 2.0,
        apy: 4.5,
        value: 5000,
        rewards: 125.50
      },
      {
        id: '2',
        protocol: 'Uniswap V3',
        type: 'liquidity',
        asset: 'ETH/USDC',
        amount: 1.5,
        apy: 12.8,
        value: 3750,
        rewards: 480.25
      },
      {
        id: '3',
        protocol: 'Aave',
        type: 'staking',
        asset: 'AAVE',
        amount: 50,
        apy: 8.2,
        value: 7500,
        rewards: 615.75
      }
    ];
    
    // Mock transactions
    const mockTransactions: BlockchainTransaction[] = [
      {
        hash: '0x1a2b3c4d5e6f7890abcdef1234567890abcdef1234567890abcdef1234567890',
        type: 'send',
        amount: 0.5,
        currency: 'ETH',
        from: '0x742d35Cc7Da82a2C4C1C24781C3C90C4AEa3b1C5',
        to: '0x123456789abcdef123456789abcdef123456789',
        timestamp: new Date(Date.now() - 3600000),
        status: 'confirmed',
        gasUsed: 21000,
        gasFee: 0.002
      },
      {
        hash: '0x2b3c4d5e6f7890abcdef1234567890abcdef1234567890abcdef1234567890ab',
        type: 'mint',
        amount: 1,
        currency: 'NFT',
        from: '0x0000000000000000000000000000000000000000',
        to: '0x742d35Cc7Da82a2C4C1C24781C3C90C4AEa3b1C5',
        timestamp: new Date(Date.now() - 7200000),
        status: 'confirmed',
        gasUsed: 85000,
        gasFee: 0.008
      },
      {
        hash: '0x3c4d5e6f7890abcdef1234567890abcdef1234567890abcdef1234567890abcd',
        type: 'swap',
        amount: 100,
        currency: 'USDC',
        from: '0x742d35Cc7Da82a2C4C1C24781C3C90C4AEa3b1C5',
        to: '0xUniswapV3Router',
        timestamp: new Date(Date.now() - 10800000),
        status: 'pending',
        gasUsed: 0,
        gasFee: 0.015
      }
    ];
    
    setNftAssets(mockNFTs);
    setDeFiPositions(mockDeFi);
    setTransactions(mockTransactions);
    
    // Calculate portfolio value
    const totalValue = mockDeFi.reduce((sum, position) => sum + position.value, 0);
    setPortfolioValue(totalValue);
  };

  const loadUserAssets = async (address: string) => {
    // In a real implementation, this would fetch from blockchain APIs
    loadMockData();
  };

  const startBlockchainVisualization = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const blocks: Array<{ x: number; y: number; size: number; hash: string; timestamp: number }> = [];
    
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Add new block occasionally
      if (Math.random() < 0.02) {
        blocks.push({
          x: Math.random() * canvas.width,
          y: canvas.height + 20,
          size: 20 + Math.random() * 20,
          hash: Math.random().toString(36).substring(2, 10),
          timestamp: Date.now()
        });
      }
      
      // Update and draw blocks
      for (let i = blocks.length - 1; i >= 0; i--) {
        const block = blocks[i];
        block.y -= 1;
        
        // Remove blocks that are off screen
        if (block.y + block.size < 0) {
          blocks.splice(i, 1);
          continue;
        }
        
        // Draw block
        const gradient = ctx.createRadialGradient(
          block.x, block.y, 0,
          block.x, block.y, block.size
        );
        gradient.addColorStop(0, '#00FFFF');
        gradient.addColorStop(1, 'transparent');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(block.x - block.size/2, block.y - block.size/2, block.size, block.size);
        
        // Draw connections
        if (i < blocks.length - 1) {
          const nextBlock = blocks[i + 1];
          ctx.strokeStyle = 'rgba(0, 255, 255, 0.3)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(block.x, block.y);
          ctx.lineTo(nextBlock.x, nextBlock.y);
          ctx.stroke();
        }
      }
      
      requestAnimationFrame(animate);
    };
    
    animate();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'confirmed': return '#00FF00';
      case 'pending': return '#FFFF00';
      case 'failed': return '#FF0040';
      default: return '#666666';
    }
  };

  const formatAddress = (address: string) => {
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const formatHash = (hash: string) => {
    return `${hash.slice(0, 10)}...${hash.slice(-8)}`;
  };

  return (
    <div style={{
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)',
      color: '#00FFFF',
      fontFamily: 'Orbitron, monospace',
      minHeight: '100vh',
      padding: '20px',
      position: 'relative'
    }}>
      {/* Blockchain Network Background */}
      <canvas
        ref={canvasRef}
        width={1200}
        height={800}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          opacity: 0.1,
          zIndex: 0
        }}
      />

      {/* Header */}
      <motion.header
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '30px',
          zIndex: 10,
          position: 'relative'
        }}
      >
        <div>
          <h1 style={{
            fontSize: '3rem',
            margin: 0,
            background: 'linear-gradient(45deg, #00FFFF, #FF00FF, #FFFF00)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textShadow: '0 0 20px #00FFFF',
            animation: 'web3Glow 3s ease-in-out infinite alternate'
          }}>
            WEB3 INTEGRATION
          </h1>
          <div style={{
            display: 'flex',
            gap: '20px',
            marginTop: '10px',
            fontSize: '0.9rem'
          }}>
            <span>Network: {currentNetwork.toUpperCase()}</span>
            {wallet && <span>Portfolio: ${portfolioValue.toLocaleString()}</span>}
            <span style={{ color: marketData.eth.change24h > 0 ? '#00FF00' : '#FF0040' }}>
              ETH: ${marketData.eth.price} ({marketData.eth.change24h > 0 ? '+' : ''}{marketData.eth.change24h}%)
            </span>
          </div>
        </div>
        
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          {!wallet ? (
            <button
              onClick={() => connectWallet()}
              disabled={isConnecting}
              style={{
                padding: '12px 25px',
                background: 'linear-gradient(45deg, #FF00FF, #FFFF00)',
                border: 'none',
                borderRadius: '8px',
                color: '#000',
                fontWeight: 'bold',
                cursor: 'pointer',
                fontSize: '1rem'
              }}
            >
              {isConnecting ? 'CONNECTING...' : '🔒 CONNECT WALLET'}
            </button>
          ) : (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '15px',
              padding: '12px 20px',
              background: 'rgba(0,0,0,0.8)',
              border: '2px solid #00FF00',
              borderRadius: '12px',
              backdropFilter: 'blur(10px)'
            }}>
              <div>
                <div style={{ fontSize: '0.9rem', opacity: 0.8 }}>Connected: {wallet.provider}</div>
                <div style={{ fontWeight: 'bold' }}>{formatAddress(wallet.address)}</div>
                <div style={{ color: '#FFFF00' }}>{wallet.balance.toFixed(4)} ETH</div>
              </div>
              <button
                onClick={disconnectWallet}
                style={{
                  padding: '8px 12px',
                  background: 'rgba(255,0,64,0.2)',
                  border: '1px solid #FF0040',
                  borderRadius: '6px',
                  color: '#FF0040',
                  cursor: 'pointer'
                }}
              >
                Disconnect
              </button>
            </div>
          )}
        </div>
      </motion.header>

      {wallet && (
        <>
          {/* Portfolio Overview */}
          <motion.section
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '20px',
              marginBottom: '30px',
              zIndex: 10,
              position: 'relative'
            }}
          >
            {[
              { label: 'Total Portfolio', value: `$${portfolioValue.toLocaleString()}`, color: '#00FFFF', icon: '💰' },
              { label: 'NFT Collection', value: nftAssets.length.toString(), color: '#FF00FF', icon: '🖼️' },
              { label: 'DeFi Positions', value: defiPositions.length.toString(), color: '#FFFF00', icon: '🌱' },
              { label: 'Total Rewards', value: `$${defiPositions.reduce((sum, p) => sum + p.rewards, 0).toFixed(2)}`, color: '#00FF00', icon: '🎆' }
            ].map((metric, index) => (
              <div
                key={metric.label}
                style={{
                  background: 'rgba(0,0,0,0.8)',
                  border: `2px solid ${metric.color}`,
                  borderRadius: '15px',
                  padding: '25px',
                  textAlign: 'center',
                  position: 'relative',
                  overflow: 'hidden',
                  boxShadow: `0 0 30px rgba(${metric.color === '#00FFFF' ? '0,255,255' : metric.color === '#FF00FF' ? '255,0,255' : metric.color === '#FFFF00' ? '255,255,0' : '0,255,0'},0.3)`
                }}
              >
                <div style={{ fontSize: '3rem', marginBottom: '15px' }}>{metric.icon}</div>
                <div style={{
                  fontSize: '2.5rem',
                  fontWeight: 'bold',
                  color: metric.color,
                  marginBottom: '8px'
                }}>
                  {metric.value}
                </div>
                <div style={{ fontSize: '1rem', opacity: 0.8 }}>{metric.label}</div>
                
                {/* Animated border effect */}
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: '-100%',
                  right: 0,
                  height: '2px',
                  background: `linear-gradient(90deg, transparent, ${metric.color}, transparent)`,
                  animation: 'web3Scan 3s linear infinite'
                }} />
              </div>
            ))}
          </motion.section>

          {/* NFT Gallery */}
          <motion.section
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            style={{
              background: 'rgba(0,0,0,0.8)',
              border: '2px solid #FF00FF',
              borderRadius: '15px',
              padding: '30px',
              marginBottom: '30px',
              position: 'relative',
              zIndex: 10
            }}
          >
            <h2 style={{ color: '#FF00FF', marginBottom: '25px' }}>NFT COLLECTION</h2>
            
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '20px'
            }}>
              {nftAssets.map((nft, index) => (
                <motion.div
                  key={nft.id}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.5 + index * 0.1 }}
                  whileHover={{ scale: 1.05, y: -10 }}
                  style={{
                    background: 'rgba(0,0,0,0.9)',
                    border: '2px solid #FF00FF',
                    borderRadius: '12px',
                    padding: '20px',
                    cursor: 'pointer',
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                  onClick={() => setSelectedNFT(nft)}
                >
                  <img
                    src={nft.image}
                    alt={nft.name}
                    style={{
                      width: '100%',
                      height: '200px',
                      objectFit: 'cover',
                      borderRadius: '8px',
                      marginBottom: '15px',
                      border: '1px solid #333'
                    }}
                  />
                  
                  <h3 style={{ color: '#00FFFF', marginBottom: '8px' }}>{nft.name}</h3>
                  <p style={{ fontSize: '0.9rem', opacity: 0.8, marginBottom: '12px' }}>
                    {nft.description.substring(0, 80)}...
                  </p>
                  
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '10px'
                  }}>
                    <span style={{ color: '#FFFF00' }}>{nft.collection}</span>
                    <span style={{ color: '#00FF00', fontWeight: 'bold' }}>
                      {nft.price} {nft.currency}
                    </span>
                  </div>
                  
                  <div style={{
                    padding: '8px 12px',
                    background: 'rgba(255,0,255,0.2)',
                    border: '1px solid #FF00FF',
                    borderRadius: '6px',
                    fontSize: '0.8rem',
                    textAlign: 'center'
                  }}>
                    {nft.metadata.rarity}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.section>

          {/* DeFi Positions */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            style={{
              background: 'rgba(0,0,0,0.8)',
              border: '2px solid #FFFF00',
              borderRadius: '15px',
              padding: '30px',
              marginBottom: '30px',
              position: 'relative',
              zIndex: 10
            }}
          >
            <h2 style={{ color: '#FFFF00', marginBottom: '25px' }}>DeFi POSITIONS</h2>
            
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '20px'
            }}>
              {defiPositions.map((position, index) => (
                <motion.div
                  key={position.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7 + index * 0.1 }}
                  style={{
                    background: 'rgba(0,0,0,0.9)',
                    border: '2px solid #FFFF00',
                    borderRadius: '12px',
                    padding: '25px',
                    position: 'relative'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '15px' }}>
                    <div>
                      <h3 style={{ color: '#00FFFF', margin: 0 }}>{position.protocol}</h3>
                      <span style={{
                        padding: '3px 8px',
                        background: 'rgba(255,255,0,0.2)',
                        border: '1px solid #FFFF00',
                        borderRadius: '4px',
                        fontSize: '0.8rem',
                        textTransform: 'capitalize'
                      }}>
                        {position.type}
                      </span>
                    </div>
                    <div style={{
                      fontSize: '1.5rem',
                      fontWeight: 'bold',
                      color: '#00FF00'
                    }}>
                      {position.apy}% APY
                    </div>
                  </div>
                  
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '15px',
                    marginBottom: '15px'
                  }}>
                    <div>
                      <div style={{ fontSize: '0.9rem', opacity: 0.7 }}>Amount</div>
                      <div style={{ color: '#FFFF00', fontWeight: 'bold' }}>
                        {position.amount} {position.asset}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.9rem', opacity: 0.7 }}>Value</div>
                      <div style={{ color: '#00FFFF', fontWeight: 'bold' }}>
                        ${position.value.toLocaleString()}
                      </div>
                    </div>
                  </div>
                  
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '12px',
                    background: 'rgba(0,255,0,0.1)',
                    border: '1px solid #00FF00',
                    borderRadius: '8px'
                  }}>
                    <span>Rewards Earned</span>
                    <span style={{ color: '#00FF00', fontWeight: 'bold' }}>
                      ${position.rewards.toFixed(2)}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.section>

          {/* Transaction History */}
          <motion.section
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            style={{
              background: 'rgba(0,0,0,0.8)',
              border: '2px solid #00FF00',
              borderRadius: '15px',
              padding: '30px',
              position: 'relative',
              zIndex: 10
            }}
          >
            <h2 style={{ color: '#00FF00', marginBottom: '25px' }}>TRANSACTION HISTORY</h2>
            
            <div style={{
              maxHeight: '400px',
              overflowY: 'auto',
              border: '1px solid #333',
              borderRadius: '8px',
              background: 'rgba(0,0,0,0.5)'
            }}>
              {transactions.map((tx, index) => (
                <motion.div
                  key={tx.hash}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.9 + index * 0.05 }}
                  style={{
                    padding: '20px',
                    borderBottom: '1px solid #333',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
                      <span style={{
                        padding: '4px 8px',
                        background: getStatusColor(tx.status),
                        color: '#000',
                        borderRadius: '4px',
                        fontSize: '0.8rem',
                        fontWeight: 'bold',
                        textTransform: 'uppercase'
                      }}>
                        {tx.status}
                      </span>
                      <span style={{ color: '#FFFF00', textTransform: 'uppercase' }}>{tx.type}</span>
                      <span style={{ fontSize: '0.9rem', opacity: 0.7 }}>
                        {tx.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    
                    <div style={{ marginBottom: '5px' }}>
                      <span style={{ color: '#00FFFF', fontWeight: 'bold' }}>
                        {tx.amount} {tx.currency}
                      </span>
                      {tx.type !== 'mint' && (
                        <span style={{ fontSize: '0.9rem', opacity: 0.7, marginLeft: '10px' }}>
                          {formatAddress(tx.from)} → {formatAddress(tx.to)}
                        </span>
                      )}
                    </div>
                    
                    <div style={{ fontSize: '0.8rem', opacity: 0.6 }}>
                      Hash: {formatHash(tx.hash)}
                    </div>
                  </div>
                  
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ color: '#FF00FF', fontWeight: 'bold' }}>
                      Gas: {tx.gasFee.toFixed(4)} ETH
                    </div>
                    {tx.gasUsed > 0 && (
                      <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>
                        {tx.gasUsed.toLocaleString()} units
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.section>
        </>
      )}

      {/* Web3 Features Status */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.0 }}
        style={
        {
          position: 'fixed',
          bottom: '20px',
          left: '20px',
          background: 'rgba(0,0,0,0.9)',
          border: '2px solid #00FFFF',
          borderRadius: '12px',
          padding: '20px',
          minWidth: '280px',
          zIndex: 1000
        }}
      >
        <h4 style={{ color: '#00FFFF', marginBottom: '15px' }}>WEB3 FEATURES</h4>
        {Object.entries(web3Features).map(([feature, enabled]) => (
          <div key={feature} style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '10px',
            fontSize: '0.9rem'
          }}>
            <span style={{ textTransform: 'capitalize' }}>
              {feature.replace(/([A-Z])/g, ' $1').trim()}
            </span>
            <span style={{
              color: enabled ? '#00FF00' : '#666',
              fontWeight: 'bold'
            }}>
              {enabled ? '✓ ON' : '✗ OFF'}
            </span>
          </div>
        ))}
      </motion.div>

      <style jsx>{`
        @keyframes web3Glow {
          0% { filter: hue-rotate(0deg) brightness(1); }
          100% { filter: hue-rotate(120deg) brightness(1.2); }
        }
        
        @keyframes web3Scan {
          0% { transform: translateX(0); }
          100% { transform: translateX(200%); }
        }
      `}</style>
    </div>
  );
};

export default Web3Integration;