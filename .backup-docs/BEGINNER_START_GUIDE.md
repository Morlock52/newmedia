# 🎬 **COMPLETE BEGINNER'S GUIDE - How to Run Your Media Server**

## **📋 What You Need First**

1. **Docker Desktop** - Make sure it's installed and running
2. **Terminal/Command Prompt** - We'll use this to run commands
3. **5 minutes** - That's all it takes!

---

## **🚀 SUPER SIMPLE 3-STEP PROCESS**

### **Step 1: Open Terminal**
- **On Mac**: Press `Cmd + Space`, type "Terminal", press Enter
- **On Windows**: Press `Win + R`, type "cmd", press Enter

### **Step 2: Navigate to Your Project**
Copy and paste this command (replace with your actual path):
```bash
cd /Users/morlock/fun/newmedia
```

### **Step 3: Run the Magic Command**
Copy and paste this:
```bash
./deploy-optimized.sh
```

**That's it!** The script will do everything automatically.

---

## **🎯 WHAT TO EXPECT**

When you run the command, you'll see:
```
🎬 Media Server Stack 2025 - Optimized Deployment
==================================================

🛑 Stopping current deployment...
🧹 Cleaning up networks...
🚀 Deploying optimized media server stack...
⏳ Waiting for services to initialize...
✅ Deployment Complete!
```

---

## **🌐 HOW TO ACCESS YOUR SERVICES**

After deployment, you can access everything through your web browser:

### **🎬 Main Media Services**
- **Jellyfin (Netflix-like)**: http://localhost:8096
- **AudioBooks**: http://localhost:13378  
- **Music Player**: http://localhost:4533
- **Photos**: http://localhost:2283

### **📥 Download Management**
- **Torrent Downloads**: http://localhost:8080
- **Movie Manager**: http://localhost:7878
- **TV Show Manager**: http://localhost:8989

### **📊 Dashboard & Monitoring**
- **Beautiful Dashboard**: Open the file `service-access-optimized.html` in your browser
- **System Monitoring**: http://localhost:3000

---

## **🔐 LOGIN INFORMATION**

Your passwords are stored in a file called `.generated_passwords.txt`

To see your passwords:
```bash
cat .generated_passwords.txt
```

**Username for most services**: `admin`

---

## **❓ TROUBLESHOOTING**

### **If you get "permission denied":**
```bash
chmod +x deploy-optimized.sh
./deploy-optimized.sh
```

### **If Docker isn't running:**
1. Open Docker Desktop app
2. Wait for it to start (whale icon in system tray)
3. Try the command again

### **If a service isn't working:**
```bash
docker compose -f docker-compose-optimized.yml ps
```
This shows the status of all services.

---

## **🎉 WHAT YOU'LL HAVE RUNNING**

After setup, you'll have a complete media server with:
- ✅ **Netflix-like streaming** (Jellyfin)
- ✅ **Automatic movie/TV downloads**
- ✅ **Music streaming**
- ✅ **Audiobook player** 
- ✅ **Photo management**
- ✅ **Beautiful web dashboard**
- ✅ **System monitoring**

---

## **🆘 NEED HELP?**

### **Quick Commands:**
```bash
# Check if everything is running
docker compose -f docker-compose-optimized.yml ps

# Stop everything
docker compose -f docker-compose-optimized.yml down

# Start everything again  
./deploy-optimized.sh

# See your passwords
cat .generated_passwords.txt
```

### **Common Issues:**
1. **"Command not found"** → Make sure you're in the right folder
2. **"Permission denied"** → Run: `chmod +x deploy-optimized.sh`
3. **"Docker not running"** → Start Docker Desktop app
4. **Service won't load** → Wait 2-3 minutes, services take time to start

---

## **📱 PRO TIPS**

1. **Bookmark these URLs** in your browser for easy access
2. **The dashboard** (`service-access-optimized.html`) is mobile-friendly
3. **Wait 2-3 minutes** after deployment for all services to fully start
4. **Your data is safe** - everything is stored in Docker volumes

---

**🎬 Enjoy your new media server! It's now running with enterprise-grade security and modern features!**