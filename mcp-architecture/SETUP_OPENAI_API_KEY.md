# 🔑 How to Connect Your OpenAI API Key to Ultimate Media Server 2025

## Step 1: Get Your OpenAI API Key

1. **Go to OpenAI Platform**:
   - Visit: https://platform.openai.com/api-keys
   - Sign in with your OpenAI account

2. **Create a New API Key**:
   - Click "Create new secret key"
   - Give it a name like "Ultimate Media Server"
   - Copy the key (starts with `sk-...`)
   - **IMPORTANT**: Save it somewhere safe - you can't see it again!

## Step 2: Add API Key to Your Media Server

### Method 1: Edit the .env File (Recommended)

1. **Open the .env file**:
   ```bash
   cd /Users/morlock/fun/newmedia/mcp-architecture
   nano .env
   ```

2. **Find this line**:
   ```
   OPENAI_API_KEY=test-key-for-local-development
   ```

3. **Replace with your actual key**:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

4. **Save the file**:
   - Press `Ctrl+X`
   - Press `Y` to confirm
   - Press `Enter`

### Method 2: Use the Dashboard API Key Manager

1. **Open the Dashboard**:
   - Go to: http://localhost:8090
   - Look for the "🔑 API Configuration" section

2. **Click "Set OpenAI API Key"**

3. **Paste your key and save**

## Step 3: Restart the MCP Server

After adding your API key, restart the service:

```bash
docker-compose -f docker-compose.simple.yml restart mcp-dashboard
```

## Step 4: Verify It's Working

1. **Check MCP Health**:
   ```bash
   curl http://localhost:3000/health
   ```

2. **Test AI Features**:
   - Go to the dashboard: http://localhost:8090
   - Try the AI features in the MCP section

## 🤖 How to Use MCP with OpenAI

### What MCP Does with Your OpenAI API Key:

1. **Smart Content Recommendations**
   - Analyzes your viewing habits
   - Suggests new content based on preferences
   - Creates personalized playlists

2. **Automated Organization**
   - AI-powered file naming
   - Smart category detection
   - Metadata enhancement

3. **Natural Language Control**
   - Ask: "Download the latest episode of my favorite show"
   - Say: "Find me action movies from the 90s"
   - Request: "Set up a kids-friendly movie list"

### Available AI Commands in the Dashboard:

1. **Content Discovery**:
   - "Find movies similar to [movie name]"
   - "Recommend TV shows based on my history"
   - "What's trending this week?"

2. **Smart Management**:
   - "Organize my movie collection by genre"
   - "Clean up duplicate downloads"
   - "Optimize my library metadata"

3. **Automation**:
   - "Schedule downloads for off-peak hours"
   - "Auto-download subtitles in Spanish"
   - "Set up quality preferences for 4K content"

## 📊 MCP Tools Available with OpenAI:

| Tool | What It Does |
|------|--------------|
| `searchContent` | AI-powered content search across all services |
| `recommendMedia` | Get personalized recommendations |
| `analyzeLibrary` | AI analysis of your media collection |
| `optimizeSettings` | AI-suggested optimal settings |
| `naturalLanguageControl` | Control services with plain English |
| `trendAnalysis` | Track and predict content trends |
| `qualityOptimizer` | AI-based quality/size optimization |
| `metadataEnhancer` | Enrich media metadata using AI |
| `scheduleManager` | Intelligent download scheduling |
| `contentModerator` | Family-friendly content filtering |

## 🎯 Quick Test

Once your API key is set up, try this:

1. **Open Dashboard**: http://localhost:8090
2. **Go to MCP Section**: Look for the AI Control panel
3. **Try a Command**: Type "Show me popular movies this week"
4. **Check Response**: You should get AI-powered results!

## 💰 API Usage Tips

- **Free Tier**: OpenAI offers free credits for new accounts
- **Costs**: Most queries cost fractions of a cent
- **Optimization**: MCP caches responses to minimize API calls
- **Budget Control**: Set monthly limits in OpenAI dashboard

## 🔧 Troubleshooting

If the API key isn't working:

1. **Check the logs**:
   ```bash
   docker-compose -f docker-compose.simple.yml logs mcp-dashboard | grep -i openai
   ```

2. **Verify key format**:
   - Should start with `sk-`
   - No extra spaces or quotes
   - Exact copy from OpenAI

3. **Test directly**:
   ```bash
   curl http://localhost:3000/tools/searchContent \
     -H "Content-Type: application/json" \
     -d '{"query": "test"}'
   ```

## 🚀 Advanced Features

Once your API key is working, you can:

1. **Enable Voice Control**: Use voice commands to control your media server
2. **Smart Playlists**: AI-generated playlists based on mood/activity
3. **Predictive Downloads**: AI predicts what you'll want to watch
4. **Content Translation**: Auto-translate subtitles and descriptions
5. **Social Integration**: Share recommendations with friends

## Need Help?

- **OpenAI Documentation**: https://platform.openai.com/docs
- **MCP API Reference**: http://localhost:3000/docs
- **Dashboard Help**: Click the "?" icon in the dashboard

Your Ultimate Media Server is now AI-powered! 🎉