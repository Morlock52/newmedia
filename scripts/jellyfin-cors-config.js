/**
 * Jellyfin CORS Configuration Helper
 * Configures proper CORS settings for dashboard integration
 */

const fs = require('fs').promises;
const path = require('path');
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

class JellyfinCORSConfig {
    constructor() {
        this.jellyfinContainer = 'jellyfin';
        this.jellyfinUrl = 'http://localhost:8096';
        this.configPath = '/config/config';
    }

    /**
     * Configure CORS settings for Jellyfin
     */
    async configureCORS() {
        console.log('🔧 Configuring Jellyfin CORS settings...');

        try {
            // Create network configuration with CORS settings
            const networkConfig = `<?xml version="1.0" encoding="utf-8"?>
<NetworkConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <EnableHttps>false</EnableHttps>
  <RequireHttps>false</RequireHttps>
  <HttpServerPortNumber>8096</HttpServerPortNumber>
  <HttpsPortNumber>8920</HttpsPortNumber>
  <EnableRemoteAccess>true</EnableRemoteAccess>
  <EnableAutomaticPortForwarding>false</EnableAutomaticPortForwarding>
  <KnownProxies />
  <LocalNetworkSubnets>
    <string>10.0.0.0/8</string>
    <string>172.16.0.0/12</string>
    <string>192.168.0.0/16</string>
    <string>127.0.0.1/32</string>
    <string>0.0.0.0/0</string>
  </LocalNetworkSubnets>
  <LocalNetworkAddresses />
  <EnableIPV6>false</EnableIPV6>
  <EnableIPV4>true</EnableIPV4>
  <EnablePublishedServerUriByRequest>false</EnablePublishedServerUriByRequest>
  <PublishedServerUriBySubnet />
  <RemoteIPFilter />
  <IsRemoteIPFilterBlacklist>false</IsRemoteIPFilterBlacklist>
  <EnableUPnP>false</EnableUPnP>
  <CertificatePath />
  <CertificatePassword />
  <BaseUrl />
</NetworkConfiguration>`;

            // Write network configuration
            await execAsync(`docker exec ${this.jellyfinContainer} bash -c 'mkdir -p ${this.configPath} && cat > ${this.configPath}/network.xml << "EOF"
${networkConfig}
EOF'`);

            console.log('✅ Network configuration updated');

            // Create CORS policy configuration
            const corsConfig = `<?xml version="1.0" encoding="utf-8"?>
<CorsConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <CorsPolicy>
    <Origins>
      <string>http://localhost:3000</string>
      <string>http://localhost:3001</string>
      <string>http://localhost:3002</string>
      <string>http://localhost:3003</string>
      <string>http://localhost:5000</string>
      <string>http://localhost:8000</string>
      <string>http://localhost:8080</string>
      <string>http://127.0.0.1:3000</string>
      <string>http://127.0.0.1:3001</string>
      <string>http://127.0.0.1:3002</string>
      <string>http://127.0.0.1:3003</string>
      <string>*</string>
    </Origins>
    <Methods>
      <string>GET</string>
      <string>POST</string>
      <string>PUT</string>
      <string>DELETE</string>
      <string>OPTIONS</string>
    </Methods>
    <Headers>
      <string>*</string>
      <string>Authorization</string>
      <string>Content-Type</string>
      <string>X-Emby-Token</string>
      <string>X-MediaBrowser-Token</string>
    </Headers>
    <AllowCredentials>true</AllowCredentials>
    <PreflightMaxAge>86400</PreflightMaxAge>
  </CorsPolicy>
</CorsConfiguration>`;

            await execAsync(`docker exec ${this.jellyfinContainer} bash -c 'cat > ${this.configPath}/cors.xml << "EOF"
${corsConfig}
EOF'`);

            console.log('✅ CORS configuration updated');

            return true;
        } catch (error) {
            console.error('❌ Error configuring CORS:', error.message);
            return false;
        }
    }

    /**
     * Update system configuration for better API access
     */
    async updateSystemConfig() {
        console.log('🔧 Updating system configuration...');

        try {
            const systemConfig = `<?xml version="1.0" encoding="utf-8"?>
<ServerConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <LogFileRetentionDays>3</LogFileRetentionDays>
  <IsStartupWizardCompleted>true</IsStartupWizardCompleted>
  <EnableMetrics>false</EnableMetrics>
  <EnableNormalizedItemByNameIds>true</EnableNormalizedItemByNameIds>
  <IsPortAuthorized>true</IsPortAuthorized>
  <QuickConnectAvailable>true</QuickConnectAvailable>
  <EnableCaseSensitiveItemIds>true</EnableCaseSensitiveItemIds>
  <DisableLiveTvChannelUserDataName>true</DisableLiveTvChannelUserDataName>
  <MetadataPath />
  <PreferredMetadataLanguage>en</PreferredMetadataLanguage>
  <MetadataCountryCode>US</MetadataCountryCode>
  <RemoteClientBitrateLimit>0</RemoteClientBitrateLimit>
  <EnableFolderView>false</EnableFolderView>
  <EnableGroupingIntoCollections>false</EnableGroupingIntoCollections>
  <DisplaySpecialsWithinSeasons>true</DisplaySpecialsWithinSeasons>
  <LocalNetworkSubnets>
    <string>10.0.0.0/8</string>
    <string>172.16.0.0/12</string>
    <string>192.168.0.0/16</string>
    <string>127.0.0.1/32</string>
    <string>0.0.0.0/0</string>
  </LocalNetworkSubnets>
  <EnableExternalContentInSuggestions>true</EnableExternalContentInSuggestions>
  <ImageExtractionTimeoutMs>0</ImageExtractionTimeoutMs>
  <PathSubstitutions />
  <UninstalledPlugins />
  <CollapseVideoFolders>false</CollapseVideoFolders>
  <EnablePeoplePrefixSubFolders>true</EnablePeoplePrefixSubFolders>
  <UICulture>en-US</UICulture>
  <SaveMetadataHidden>false</SaveMetadataHidden>
  <ContentTypes />
  <RemoteClientBitrateLimit>0</RemoteClientBitrateLimit>
  <EnableDashboard>true</EnableDashboard>
  <EnableThumbnailsForRemoteItems>true</EnableThumbnailsForRemoteItems>
  <EnableSlowResponseWarning>false</EnableSlowResponseWarning>
  <EnableDebugLevelLogging>false</EnableDebugLevelLogging>
  <EnableAutoRunWebApp>false</EnableAutoRunWebApp>
</ServerConfiguration>`;

            await execAsync(`docker exec ${this.jellyfinContainer} bash -c 'cat > ${this.configPath}/system.xml << "EOF"
${systemConfig}
EOF'`);

            console.log('✅ System configuration updated');
            return true;
        } catch (error) {
            console.error('❌ Error updating system config:', error.message);
            return false;
        }
    }

    /**
     * Test API connectivity
     */
    async testAPI() {
        console.log('🧪 Testing API connectivity...');

        const endpoints = [
            '/System/Info',
            '/System/Configuration',
            '/System/Ping',
            '/health'
        ];

        const results = [];

        for (const endpoint of endpoints) {
            try {
                const { exec } = require('child_process');
                const { promisify } = require('util');
                const execAsync = promisify(exec);

                await execAsync(`curl -s --connect-timeout 5 "${this.jellyfinUrl}${endpoint}" > /dev/null`);
                results.push({ endpoint, status: '✅ OK' });
            } catch (error) {
                results.push({ endpoint, status: '❌ Failed' });
            }
        }

        console.log('\n📊 API Test Results:');
        results.forEach(result => {
            console.log(`  ${result.endpoint} - ${result.status}`);
        });

        return results;
    }

    /**
     * Restart Jellyfin container
     */
    async restartContainer() {
        console.log('🔄 Restarting Jellyfin container...');

        try {
            await execAsync(`docker restart ${this.jellyfinContainer}`);
            
            // Wait for container to be ready
            console.log('⏳ Waiting for Jellyfin to start...');
            await new Promise(resolve => setTimeout(resolve, 15000));

            console.log('✅ Container restarted successfully');
            return true;
        } catch (error) {
            console.error('❌ Error restarting container:', error.message);
            return false;
        }
    }

    /**
     * Run full configuration process
     */
    async configure() {
        console.log('🚀 Starting Jellyfin CORS configuration...\n');

        const steps = [
            { name: 'Configure CORS', method: () => this.configureCORS() },
            { name: 'Update System Config', method: () => this.updateSystemConfig() },
            { name: 'Restart Container', method: () => this.restartContainer() },
            { name: 'Test API', method: () => this.testAPI() }
        ];

        for (const step of steps) {
            console.log(`\n📋 ${step.name}...`);
            const success = await step.method();
            
            if (!success) {
                console.log(`❌ ${step.name} failed`);
                return false;
            }
        }

        console.log('\n🎉 Jellyfin CORS configuration completed successfully!');
        console.log(`🌐 Access Jellyfin at: ${this.jellyfinUrl}`);
        
        return true;
    }
}

// Export for use in other modules
module.exports = JellyfinCORSConfig;

// Run if called directly
if (require.main === module) {
    const config = new JellyfinCORSConfig();
    config.configure()
        .then(success => {
            process.exit(success ? 0 : 1);
        })
        .catch(error => {
            console.error('❌ Configuration failed:', error);
            process.exit(1);
        });
}