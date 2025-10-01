# 🚂 Railway Deployment Guide - Keep Your Workflow

This guide helps you deploy your **exact docker-compose workflow** to Railway + n8n Cloud.

---

## 🎯 What We're Deploying

**Current Setup (localhost):**
```
Docker Compose:
├── Tagging API (port 5001) ← Your Flask API
└── n8n (port 5678) ← Workflow automation (calls your API)
```

**New Setup (TWO separate cloud services):**
```
Railway (hosts your API):
└── Tagging API → Deploy from GitHub ← YOUR PYTHON CODE

n8n Cloud (hosts workflows):  
└── Your workflows → Calls Railway API via HTTP ← AUTOMATION ONLY
```

**Important:** These are **separate deployments**:
1. **Railway** = Your custom API code (Flask app)
2. **n8n Cloud** = Pre-built workflow tool (not your code)

You do NOT deploy n8n with your code. n8n Cloud is a separate service that connects to your Railway API.

---

## 📝 Step-by-Step Guide

### **Step 1: Push Your Code to GitHub**

```bash
# Make sure everything is committed
git add .
git commit -m "Prepare for Railway deployment"
git push origin main
```

### **Step 2: Deploy Your API to Railway**

**Important:** You're deploying **ONLY your API code** to Railway, NOT n8n.

1. **Go to Railway**
   - Visit: https://railway.app
   - Sign up with GitHub (free tier)

2. **Create New Project**
   - Click "New Project"
   - Select **"Deploy from GitHub repo"** ← Deploy YOUR code
   - ❌ Do NOT select "Deploy n8n" or any template
   - Choose your `unified-product-system` repo
   - Railway will auto-detect your `Dockerfile`
   
   **Why GitHub?** Because Railway needs to build and run YOUR Flask API (the tagging service).

3. **Add Environment Variables**
   - Click on your service
   - Go to "Variables" tab
   - Add these variables:
   
   ```
   OPENAI_API_KEY=sk-your-actual-openai-key-here
   FLASK_ENV=production
   ```

4. **Generate Public Domain**
   - Go to "Settings" tab
   - Scroll to "Networking"
   - Click "Generate Domain"
   - Copy the URL (e.g., `https://unified-product-system-production.up.railway.app`)
   - ⭐ **Save this URL - you'll need it for n8n!**

5. **Wait for Deployment**
   - Check "Deployments" tab
   - Wait 2-5 minutes for build to complete
   - Status should show "Active ✓"

6. **Test Your API**
   ```bash
   # Replace with your actual Railway URL
   curl https://your-app.up.railway.app/api/health
   ```
   
   Should return:
   ```json
   {
     "status": "healthy",
     "timestamp": "2025-10-01T12:00:00",
     "taxonomy_ready": false
   }
   ```

7. **Setup Taxonomy (One-Time)**
   
   ⚠️ **Important:** Your taxonomy data needs to be set up:
   
   ```bash
   curl -X POST https://your-app.up.railway.app/api/setup-taxonomy \
     -H "Content-Type: application/json" \
     -d '{"taxonomy_file": "texo/FULL TEXO.xlsx"}'
   ```
   
   This takes ~5-10 minutes. It generates:
   - Taxonomy descriptors
   - FAISS embeddings
   
   After this completes, check health again:
   ```bash
   curl https://your-app.up.railway.app/api/health
   ```
   
   Should show: `"taxonomy_ready": true`

---

### **Step 3: Set Up n8n Cloud** (Separate Service)

**Note:** n8n Cloud is a **completely separate service** from Railway. You're not deploying code here - you're using n8n's hosted service.

1. **Sign Up for n8n Cloud**
   - Visit: https://n8n.cloud
   - Create free account (sign up, not deploy)
   - You'll get a workspace URL like: `https://your-name.app.n8n.cloud`
   - This is n8n's pre-built service (not your code!)

2. **Access Your Workspace**
   - Log in to your n8n cloud workspace
   - You'll see the n8n interface (same as localhost!)
   - This replaces your local n8n container from docker-compose

3. **Import Your Workflow**
   - Click "Workflows" in sidebar
   - Click "+ Add workflow" → "Import from File"
   - Upload your workflow JSON from `n8n_workflows/` folder
   - Common workflows:
     - `complete-workflow.json` - Full pipeline
     - `unified-workflow.json` - With Google Sheets
     - `translation-only.json` - Translation only

4. **Update API URLs in Workflow**
   
   Your workflows currently point to `localhost`. Update them:
   
   - Find **HTTP Request** nodes in your workflow
   - Change URLs from:
     ```
     http://localhost:5001/api/tag-products-simple
     ```
   - To your Railway URL:
     ```
     https://your-app.up.railway.app/api/tag-products-simple
     ```

5. **Set Up Credentials (If Using Google Sheets)**
   
   - Go to "Credentials" in n8n
   - Add "Google Sheets OAuth2 API"
   - Follow the OAuth flow
   - Grant permissions:
     - Read/write to Google Sheets
     - Google Drive access

---

### **Step 4: Test Complete Workflow**

1. **Prepare Test Data**
   
   Create a Google Sheet with these columns:
   ```
   Title | Body (HTML) | Product Type
   Power Drill | Professional cordless drill with battery | Tools
   Hammer | Steel claw hammer 16oz | Hand Tools
   ```

2. **Configure Workflow**
   
   - Open your imported workflow in n8n Cloud
   - Update Google Sheets node:
     - Select your credential
     - Enter your Sheet ID
     - Map columns correctly

3. **Execute Workflow**
   
   - Click "Execute Workflow" button (top-right)
   - Watch the execution flow
   - Each node should turn green ✓
   - Check output data

4. **Check Results**
   
   Your products should now have:
   - ✅ Tags (e.g., `tools,power-tools,drills`)
   - ✅ Confidence scores
   - ✅ Taxonomy hierarchy (L1, L2, L3, L4)

---

## 🔥 Quick Test Command

Test your Railway API directly from command line:

```bash
# Set your Railway URL
export API_URL="https://your-app.up.railway.app"

# Test with sample data
curl -X POST $API_URL/api/tag-products-simple \
  -H "Content-Type: application/json" \
  -d '{
    "products_data": [
      {
        "Title": "Power Drill Set",
        "Body (HTML)": "Professional cordless drill with 2 batteries and charger",
        "Product Type": "Power Tools"
      }
    ]
  }'
```

Expected response:
```json
{
  "status": "success",
  "data": [
    {
      "Title": "Power Drill Set",
      "tags": "tools,power-tools,drills",
      "L1_category": "Tools & Equipment",
      "L2_category": "Power Tools",
      ...
    }
  ],
  "stats": {
    "total_products": 1,
    "tagged_products": 1
  }
}
```

---

## 💡 Key Differences: Local vs Cloud

| Feature | Localhost (docker-compose) | Railway + n8n Cloud |
|---------|---------------------------|---------------------|
| **API URL** | `http://localhost:5001` | `https://your-app.up.railway.app` |
| **n8n URL** | `http://localhost:5678` | `https://your-name.app.n8n.cloud` |
| **Setup** | `docker-compose up` | Deploy once, always available |
| **Cost** | Free (local resources) | Free tier: $5/month credit |
| **Access** | Only on your machine | Accessible from anywhere |
| **Data** | Local volumes | Need to setup taxonomy once |

---

## 🛠️ Troubleshooting

### Issue: "Taxonomy not ready"

**Solution:**
```bash
# Run setup-taxonomy endpoint (one-time)
curl -X POST https://your-app.up.railway.app/api/setup-taxonomy
```

### Issue: Railway deployment failed

**Check:**
1. Go to Railway → Deployments → Click on failed deployment → View Logs
2. Common issues:
   - Missing `OPENAI_API_KEY` environment variable
   - Dockerfile errors (should be fixed now)
   - Build timeout (increase timeout in Railway settings)

### Issue: n8n workflow fails

**Check:**
1. API URL is correct (no `localhost`, use Railway URL)
2. Railway service is running (check Railway dashboard)
3. Test API endpoint with curl first
4. Check n8n execution logs for error details

### Issue: "OpenAI API key not found"

**Solution:**
- Go to Railway → Your Service → Variables
- Make sure `OPENAI_API_KEY` is set correctly
- Redeploy if needed

### Issue: Slow responses or timeouts

**Causes:**
- Railway free tier has limited resources (512MB RAM)
- OpenAI API rate limits
- Large batch processing

**Solutions:**
- Reduce batch sizes in your workflow
- Process products in smaller chunks
- Consider upgrading Railway plan if needed

---

## 📊 Monitoring Your Deployment

### Railway Dashboard
- **Metrics**: View CPU, Memory, Network usage
- **Logs**: Real-time logs from your API
- **Deployments**: History of all deployments
- **Usage**: Track your $5 monthly credit

### n8n Cloud
- **Executions**: See all workflow runs
- **Logs**: Debug failed workflows
- **Webhooks**: Set up automatic triggers

---

## 💰 Staying Within Free Tier

**Railway Free Tier:**
- $5 monthly credit
- ~100-150 hours of runtime
- Should cover 50-100 product taggings/day

**Tips to Save:**
1. Process products in batches (not real-time)
2. Use n8n scheduled workflows (e.g., once per day)
3. Don't run 24/7 - service sleeps when idle
4. Monitor usage in Railway dashboard

**n8n Cloud Free Tier:**
- 5,000 workflow executions/month
- Should be plenty for a demo!

---

## ✅ Final Checklist

- [ ] Railway project created
- [ ] Environment variables set (`OPENAI_API_KEY`)
- [ ] Public domain generated
- [ ] API health check passing
- [ ] Taxonomy setup completed
- [ ] n8n Cloud account created
- [ ] Workflow imported to n8n
- [ ] API URLs updated in workflow (localhost → Railway)
- [ ] Test workflow executed successfully
- [ ] Results verified

---

## 🎉 You're Done!

Your workflow is now running in the cloud! 

**Next Steps:**
1. Create more workflows in n8n Cloud
2. Connect to your e-commerce platform
3. Automate product tagging at scale
4. Monitor usage and costs

**Need Help?**
- Railway Docs: https://docs.railway.app
- n8n Docs: https://docs.n8n.io
- Railway Discord: Active community
- n8n Community Forum: https://community.n8n.io

---

Good luck! 🚀


