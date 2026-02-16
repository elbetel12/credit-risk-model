# 🎯 QUICK START GUIDE - Credit Risk Dashboard

## ⚡ 60-Second Setup

### 1️⃣ Double-click this file:
```
start_dashboard.bat
```

### 2️⃣ Wait ~10 seconds for both services to start

### 3️⃣ Dashboard opens automatically in browser!

---

## 🎮 How to Use the Dashboard

### Step 1: Load Sample Customer
Click the **"🎯 Load Sample Customer"** button in the sidebar

### Step 2: Adjust Features (Optional)
Modify any of the 5 features:
- 📅 Days Since Last Transaction
- 🔄 Monthly Transaction Frequency  
- 📊 Amount Volatility
- 💰 Average Transaction Amount
- 📅 Weekend Transaction Ratio

### Step 3: Assess Risk
Click the **"🚀 Assess Credit Risk"** button

### Step 4: View Results
See the 3 result cards showing:
- 📊 **Risk Score** (percentage with color)
- 🎯 **Risk Category** (Low/Medium/High)
- 💵 **Credit Limit** (in UGX)

---

## 📸 Screenshot Checklist for Submission

- [ ] Welcome screen
- [ ] Input form (sidebar)
- [ ] Sample customer results (Medium Risk)
- [ ] Low risk example
- [ ] High risk example  
- [ ] API details expanded view

---

## 🎨 Color Guide

| Risk Level | Color | Score Range |
|------------|-------|-------------|
| ✅ **Low** | 🟢 Green | 0-33% |
| ⚠️ **Medium** | 🟠 Orange | 33-67% |
| 🚨 **High** | 🔴 Red | 67-100% |

---

## ⚙️ Services

Once started, you have access to:

- 🎨 **Dashboard**: http://localhost:8501
- 🔌 **API Docs**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/health

---

## 🆘 Quick Fixes

**Problem**: "Cannot connect to API"  
**Fix**: Wait 10 more seconds for API to fully start

**Problem**: Dashboard won't open  
**Fix**: Manually visit http://localhost:8501

**Problem**: Ports in use  
**Fix**: Close other terminals and restart

---

## 📊 Sample Values for Testing

### Low Risk Customer
- Recency: **5** days
- Frequency: **35** txn/month
- Volatility: **0.15**
- Avg Amount: **150,000** UGX
- Weekend Ratio: **0.10**

### High Risk Customer  
- Recency: **90** days
- Frequency: **5** txn/month
- Volatility: **0.75**
- Avg Amount: **25,000** UGX
- Weekend Ratio: **0.50**

---

**That's it! You're ready to demonstrate! 🚀**
