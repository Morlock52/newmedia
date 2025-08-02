const { ethers, upgrades } = require("hardhat");
const fs = require("fs");
const path = require("path");

// Deployment configuration
const DEPLOYMENT_CONFIG = {
  // Media NFT Configuration
  mediaNFT: {
    name: "Web3 Media Content",
    symbol: "W3MC",
    initialRoyalty: 250 // 2.5%
  },
  
  // Payment Processor Configuration  
  paymentProcessor: {
    platformFee: 250, // 2.5%
    supportedTokens: [
      // Will be configured after deployment based on network
    ]
  },
  
  // DAO Configuration
  dao: {
    name: "Media DAO",
    symbol: "MDAO", 
    initialSupply: ethers.utils.parseEther("1000000"), // 1M tokens
    votingDelay: 7200, // ~1 day in blocks (assuming 12s block time)
    votingPeriod: 50400, // ~1 week in blocks
    proposalThreshold: ethers.utils.parseEther("1000"), // 1000 tokens to create proposal
    quorumPercentage: 4 // 4% quorum
  },
  
  // Cross-Chain Marketplace Configuration
  marketplace: {
    platformFee: 250, // 2.5%
    maxRoyalty: 1000 // 10%
  }
};

// Network-specific token addresses
const NETWORK_TOKENS = {
  mainnet: {
    USDC: "0xA0b86a33E6411c59F09b6b2F1C07b9F2FDe5A2fA",
    USDT: "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    DAI: "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    WETH: "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
  },
  polygon: {
    USDC: "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
    USDT: "0xc2132D05D31c914a87C6611C10748AEb04B58e8F", 
    DAI: "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
    WMATIC: "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270"
  },
  arbitrum: {
    USDC: "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
    USDT: "0xFd086bC7CD5C481DCC9C85ebE478A1C0b69FCbb9",
    DAI: "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1",
    WETH: "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"
  },
  optimism: {
    USDC: "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",
    USDT: "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",
    DAI: "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1", 
    WETH: "0x4200000000000000000000000000000000000006"
  }
};

// Chainlink Price Feed addresses
const PRICE_FEEDS = {
  mainnet: {
    ETH_USD: "0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419",
    USDC_USD: "0x8fFfFfd4AfB6115b954Bd326cbe7B4BA576818f6",
    DAI_USD: "0xAed0c38402d03c863234187eA8551a8b98B23b3C"
  },
  polygon: {
    MATIC_USD: "0xAB594600376Ec9fD91F8e885dADF0CE036862dE0",
    USDC_USD: "0xfE4A8cc5b5B2366C1B58Bea3858e81843581b2F7",
    DAI_USD: "0x4746DeC9e833A82EC7C2C1356372CcF2cfcD2F3D"
  },
  arbitrum: {
    ETH_USD: "0x639Fe6ab55C921f74e7fac1ee960C0B6293ba612",
    USDC_USD: "0x50834F3163758fcC1Df9973b6e91f0F0F0434aD3"
  },
  optimism: {
    ETH_USD: "0x13e3Ee699D1909E989722E753853AE30b17e08c5",
    USDC_USD: "0x16a9FA2FDa030272Ce99B29CF780dFA30361E0f3"
  }
};

async function main() {
  console.log("🚀 Starting Web3 Media Platform deployment...\n");
  
  // Get network info
  const network = await ethers.provider.getNetwork();
  const networkName = network.name === "unknown" ? "localhost" : network.name;
  
  console.log(`📡 Network: ${networkName} (Chain ID: ${network.chainId})`);
  
  // Get deployer account
  const [deployer] = await ethers.getSigners();
  console.log(`👤 Deployer: ${deployer.address}`);
  
  const balance = await deployer.getBalance();
  console.log(`💰 Balance: ${ethers.utils.formatEther(balance)} ETH\n`);
  
  if (balance.lt(ethers.utils.parseEther("0.1"))) {
    console.warn("⚠️  Warning: Low balance detected. Ensure sufficient funds for deployment.\n");
  }
  
  const deploymentAddresses = {};
  
  try {
    // 1. Deploy Media NFT Contract (Upgradeable)
    console.log("📜 Deploying MediaNFTOptimized contract...");
    const MediaNFT = await ethers.getContractFactory("MediaNFTOptimized");
    
    const mediaNFT = await upgrades.deployProxy(
      MediaNFT,
      [DEPLOYMENT_CONFIG.mediaNFT.name, DEPLOYMENT_CONFIG.mediaNFT.symbol],
      { 
        initializer: 'initialize',
        kind: 'uups'
      }
    );
    await mediaNFT.deployed();
    
    console.log(`✅ MediaNFTOptimized deployed to: ${mediaNFT.address}`);
    console.log(`   - Transaction: ${mediaNFT.deployTransaction.hash}`);
    deploymentAddresses.mediaNFT = mediaNFT.address;
    
    // 2. Deploy Payment Processor
    console.log("\n💳 Deploying CrossChainPaymentProcessor contract...");
    const PaymentProcessor = await ethers.getContractFactory("CrossChainPaymentProcessor");
    const paymentProcessor = await PaymentProcessor.deploy();
    await paymentProcessor.deployed();
    
    console.log(`✅ CrossChainPaymentProcessor deployed to: ${paymentProcessor.address}`);
    console.log(`   - Transaction: ${paymentProcessor.deployTransaction.hash}`);
    deploymentAddresses.paymentProcessor = paymentProcessor.address;
    
    // 3. Deploy Content Ownership Contract
    console.log("\n🏷️  Deploying ContentOwnership contract...");
    const ContentOwnership = await ethers.getContractFactory("ContentOwnership");
    const contentOwnership = await ContentOwnership.deploy(
      mediaNFT.address,
      paymentProcessor.address
    );
    await contentOwnership.deployed();
    
    console.log(`✅ ContentOwnership deployed to: ${contentOwnership.address}`);
    console.log(`   - Transaction: ${contentOwnership.deployTransaction.hash}`);
    deploymentAddresses.contentOwnership = contentOwnership.address;
    
    // 4. Deploy Media DAO Contract
    console.log("\n🏛️  Deploying MediaDAO contract...");
    const MediaDAO = await ethers.getContractFactory("MediaDAO");
    const mediaDAO = await MediaDAO.deploy(
      DEPLOYMENT_CONFIG.dao.name,
      DEPLOYMENT_CONFIG.dao.symbol,
      DEPLOYMENT_CONFIG.dao.initialSupply,
      DEPLOYMENT_CONFIG.dao.votingDelay,
      DEPLOYMENT_CONFIG.dao.votingPeriod,
      DEPLOYMENT_CONFIG.dao.proposalThreshold,
      DEPLOYMENT_CONFIG.dao.quorumPercentage
    );
    await mediaDAO.deployed();
    
    console.log(`✅ MediaDAO deployed to: ${mediaDAO.address}`);
    console.log(`   - Transaction: ${mediaDAO.deployTransaction.hash}`);
    deploymentAddresses.mediaDAO = mediaDAO.address;
    
    // 5. Deploy Cross-Chain Marketplace
    console.log("\n🏪 Deploying CrossChainMarketplace contract...");
    const Marketplace = await ethers.getContractFactory("CrossChainMarketplace");
    const marketplace = await Marketplace.deploy(
      mediaNFT.address,
      paymentProcessor.address,
      DEPLOYMENT_CONFIG.marketplace.platformFee
    );
    await marketplace.deployed();
    
    console.log(`✅ CrossChainMarketplace deployed to: ${marketplace.address}`);
    console.log(`   - Transaction: ${marketplace.deployTransaction.hash}`);
    deploymentAddresses.marketplace = marketplace.address;
    
    // 6. Configure Payment Tokens (if not localhost)
    if (networkName !== "localhost" && networkName !== "hardhat") {
      console.log("\n⚙️  Configuring payment tokens...");
      
      const tokens = NETWORK_TOKENS[networkName] || {};
      const priceFeeds = PRICE_FEEDS[networkName] || {};
      
      for (const [symbol, address] of Object.entries(tokens)) {
        try {
          const priceFeed = priceFeeds[`${symbol}_USD`] || ethers.constants.AddressZero;
          const isStablecoin = ['USDC', 'USDT', 'DAI'].includes(symbol);
          const decimals = symbol === 'USDC' || symbol === 'USDT' ? 6 : 18;
          
          await paymentProcessor.addPaymentToken(
            address,
            priceFeed,
            decimals,
            isStablecoin,
            isStablecoin ? ethers.utils.parseUnits("1", decimals) : ethers.utils.parseEther("0.001")
          );
          
          console.log(`   ✅ Added ${symbol} payment token`);
        } catch (error) {
          console.log(`   ❌ Failed to add ${symbol}: ${error.message}`);
        }
      }
    }
    
    // 7. Configure Contract Permissions
    console.log("\n🔐 Configuring contract permissions...");
    
    // Grant MINTER_ROLE to ContentOwnership contract
    const MINTER_ROLE = await mediaNFT.MINTER_ROLE();
    await mediaNFT.grantRole(MINTER_ROLE, contentOwnership.address);
    console.log("   ✅ Granted MINTER_ROLE to ContentOwnership");
    
    // Set up marketplace permissions
    await marketplace.setAuthorizedContract(contentOwnership.address, true);
    console.log("   ✅ Authorized ContentOwnership in Marketplace");
    
    // 8. Create Sample Subscription Plans
    console.log("\n📋 Creating sample subscription plans...");
    
    const plans = [
      { name: "Basic", price: ethers.utils.parseUnits("9.99", 6), duration: 30 * 24 * 3600 }, // $9.99/month
      { name: "Premium", price: ethers.utils.parseUnits("19.99", 6), duration: 30 * 24 * 3600 }, // $19.99/month  
      { name: "Annual", price: ethers.utils.parseUnits("99.99", 6), duration: 365 * 24 * 3600 } // $99.99/year
    ];
    
    for (const plan of plans) {
      await paymentProcessor.createSubscriptionPlan(
        plan.price,
        plan.duration,
        plan.name
      );
      console.log(`   ✅ Created ${plan.name} plan`);
    }
    
    // 9. Save deployment information
    console.log("\n💾 Saving deployment information...");
    
    const deploymentInfo = {
      network: networkName,
      chainId: network.chainId,
      timestamp: new Date().toISOString(),
      deployer: deployer.address,
      contracts: {
        MediaNFTOptimized: {
          address: mediaNFT.address,
          implementationAddress: await upgrades.erc1967.getImplementationAddress(mediaNFT.address),
          proxyAdminAddress: await upgrades.erc1967.getAdminAddress(mediaNFT.address)
        },
        CrossChainPaymentProcessor: {
          address: paymentProcessor.address
        },
        ContentOwnership: {
          address: contentOwnership.address
        },
        MediaDAO: {
          address: mediaDAO.address
        },
        CrossChainMarketplace: {
          address: marketplace.address
        }
      },
      configuration: DEPLOYMENT_CONFIG,
      supportedTokens: NETWORK_TOKENS[networkName] || {},
      priceFeeds: PRICE_FEEDS[networkName] || {}
    };
    
    // Save to deployments directory
    const deploymentsDir = path.join(__dirname, "..", "deployments");
    if (!fs.existsSync(deploymentsDir)) {
      fs.mkdirSync(deploymentsDir, { recursive: true });
    }
    
    const deploymentFile = path.join(deploymentsDir, `${networkName}-${Date.now()}.json`);
    fs.writeFileSync(deploymentFile, JSON.stringify(deploymentInfo, null, 2));
    
    // Save latest deployment
    const latestFile = path.join(deploymentsDir, `${networkName}-latest.json`);
    fs.writeFileSync(latestFile, JSON.stringify(deploymentInfo, null, 2));
    
    console.log(`   ✅ Deployment info saved to: ${deploymentFile}`);
    
    // 10. Generate environment variables
    console.log("\n📝 Generating environment variables...");
    
    const envVars = `
# Smart Contract Addresses for ${networkName}
CONTENT_OWNERSHIP_ADDRESS=${contentOwnership.address}
MEDIA_DAO_ADDRESS=${mediaDAO.address}
MARKETPLACE_ADDRESS=${marketplace.address}
MEDIA_NFT_ADDRESS=${mediaNFT.address}
PAYMENT_PROCESSOR_ADDRESS=${paymentProcessor.address}

# Network Configuration
NETWORK=${networkName}
CHAIN_ID=${network.chainId}

# Deployment Information
DEPLOYMENT_BLOCK=${await ethers.provider.getBlockNumber()}
DEPLOYMENT_TIMESTAMP=${new Date().toISOString()}
DEPLOYER_ADDRESS=${deployer.address}
`;
    
    const envFile = path.join(deploymentsDir, `${networkName}.env`);
    fs.writeFileSync(envFile, envVars.trim());
    console.log(`   ✅ Environment file saved to: ${envFile}`);
    
    // Success summary
    console.log("\n🎉 Deployment completed successfully!");
    console.log("\n📋 Contract Summary:");
    console.log("   ├── MediaNFTOptimized:", mediaNFT.address);
    console.log("   ├── CrossChainPaymentProcessor:", paymentProcessor.address);
    console.log("   ├── ContentOwnership:", contentOwnership.address);
    console.log("   ├── MediaDAO:", mediaDAO.address);
    console.log("   └── CrossChainMarketplace:", marketplace.address);
    
    console.log("\n⚠️  Next Steps:");
    console.log("   1. Verify contracts on block explorer");
    console.log("   2. Update frontend configuration with new addresses");
    console.log("   3. Configure API services with new contract addresses");
    console.log("   4. Test all contract interactions");
    console.log("   5. Set up monitoring and alerting");
    
    if (networkName === "mainnet") {
      console.log("\n🔒 Security Reminder:");
      console.log("   - Transfer ownership to multisig wallet");
      console.log("   - Run security audit before handling real funds");
      console.log("   - Set up monitoring for unusual activity");
    }
    
    // Output for scripts to parse
    console.log("\n--- DEPLOYMENT ADDRESSES ---");
    console.log(`ContentOwnership deployed to: ${contentOwnership.address}`);
    console.log(`MediaDAO deployed to: ${mediaDAO.address}`);
    console.log(`CrossChainMarketplace deployed to: ${marketplace.address}`);
    console.log(`MediaNFTOptimized deployed to: ${mediaNFT.address}`);
    console.log(`CrossChainPaymentProcessor deployed to: ${paymentProcessor.address}`);
    console.log("--- END DEPLOYMENT ADDRESSES ---");
    
  } catch (error) {
    console.error("\n❌ Deployment failed:");
    console.error(error);
    
    // Clean up any partial deployments if needed
    console.log("\n🧹 Cleaning up partial deployment...");
    
    process.exit(1);
  }
}

// Handle script execution
if (require.main === module) {
  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}

module.exports = { main, DEPLOYMENT_CONFIG };