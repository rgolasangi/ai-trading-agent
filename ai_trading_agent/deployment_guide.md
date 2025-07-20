# AI Trading Agent - Google Cloud Platform Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the AI Trading Agent on Google Cloud Platform (GCP). The system is designed for high-performance, scalable options trading with comprehensive risk management.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Compute       │  │    Storage      │  │   Monitoring │ │
│  │                 │  │                 │  │              │ │
│  │ • VM Instance   │  │ • Cloud SQL     │  │ • Stackdriver│ │
│  │ • Container     │  │ • Cloud Storage │  │ • Alerting   │ │
│  │ • Auto Scaling  │  │ • Redis Cache   │  │ • Logging    │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Networking    │  │    Security     │  │   AI/ML      │ │
│  │                 │  │                 │  │              │ │
│  │ • Load Balancer │  │ • IAM           │  │ • AI Platform│ │
│  │ • VPC           │  │ • Secret Mgr    │  │ • AutoML     │ │
│  │ • Firewall      │  │ • KMS           │  │ • BigQuery   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. GCP Account Setup
- Google Cloud Platform account with billing enabled
- Project created with appropriate permissions
- gcloud CLI installed and configured

### 2. Required APIs
Enable the following APIs in your GCP project:
```bash
gcloud services enable compute.googleapis.com
gcloud services enable sql-component.googleapis.com
gcloud services enable storage-component.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

### 3. Local Development Environment
- Python 3.11+
- Node.js 20+
- Docker
- Git

## Deployment Steps

### Step 1: Environment Setup

#### 1.1 Set Environment Variables
```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export ZONE="us-central1-a"
export INSTANCE_NAME="ai-trading-agent"
export DB_INSTANCE_NAME="trading-db"
```

#### 1.2 Configure gcloud
```bash
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE
```

### Step 2: Database Setup

#### 2.1 Create Cloud SQL Instance
```bash
# Create PostgreSQL instance
gcloud sql instances create $DB_INSTANCE_NAME \
    --database-version=POSTGRES_14 \
    --tier=db-n1-standard-2 \
    --region=$REGION \
    --storage-type=SSD \
    --storage-size=100GB \
    --backup-start-time=03:00 \
    --enable-bin-log \
    --maintenance-window-day=SUN \
    --maintenance-window-hour=04

# Create database
gcloud sql databases create trading_db --instance=$DB_INSTANCE_NAME

# Create user
gcloud sql users create trading_user \
    --instance=$DB_INSTANCE_NAME \
    --password=your-secure-password
```

#### 2.2 Create Redis Instance
```bash
gcloud redis instances create trading-cache \
    --size=1 \
    --region=$REGION \
    --redis-version=redis_6_x
```

### Step 3: Storage Setup

#### 3.1 Create Cloud Storage Buckets
```bash
# Create bucket for data storage
gsutil mb gs://$PROJECT_ID-trading-data

# Create bucket for model storage
gsutil mb gs://$PROJECT_ID-trading-models

# Create bucket for logs
gsutil mb gs://$PROJECT_ID-trading-logs
```

#### 3.2 Set Bucket Permissions
```bash
# Set appropriate permissions
gsutil iam ch serviceAccount:your-service-account@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://$PROJECT_ID-trading-data
gsutil iam ch serviceAccount:your-service-account@$PROJECT_ID.iam.gserviceaccount.com:objectAdmin gs://$PROJECT_ID-trading-models
```

### Step 4: Compute Instance Setup

#### 4.1 Create VM Instance
```bash
gcloud compute instances create $INSTANCE_NAME \
    --machine-type=n1-standard-4 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=trading-agent \
    --metadata-from-file startup-script=startup-script.sh
```

#### 4.2 Create Startup Script
Create `startup-script.sh`:
```bash
#!/bin/bash

# Update system
apt-get update
apt-get upgrade -y

# Install Python 3.11
apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt-get install -y python3.11 python3.11-pip python3.11-venv

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
apt-get install -y nodejs

# Install Docker
apt-get install -y docker.io
systemctl start docker
systemctl enable docker

# Install additional dependencies
apt-get install -y git nginx supervisor postgresql-client redis-tools

# Create application directory
mkdir -p /opt/trading-agent
cd /opt/trading-agent

# Clone repository (replace with your repo)
git clone https://github.com/your-username/ai-trading-agent.git .

# Set up Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up environment variables
cat > .env << EOF
DATABASE_URL=postgresql://trading_user:your-secure-password@localhost/trading_db
REDIS_URL=redis://localhost:6379
ZERODHA_API_KEY=your-zerodha-api-key
ZERODHA_API_SECRET=your-zerodha-api-secret
ZERODHA_ACCESS_TOKEN=your-zerodha-access-token
OPENAI_API_KEY=your-openai-api-key
GCP_PROJECT_ID=$PROJECT_ID
ENVIRONMENT=production
EOF

# Set up supervisor configuration
cat > /etc/supervisor/conf.d/trading-agent.conf << EOF
[program:trading-agent]
command=/opt/trading-agent/venv/bin/python main.py
directory=/opt/trading-agent
user=root
autostart=true
autorestart=true
stderr_logfile=/var/log/trading-agent.err.log
stdout_logfile=/var/log/trading-agent.out.log
environment=PATH="/opt/trading-agent/venv/bin"
EOF

# Start services
systemctl reload supervisor
supervisorctl start trading-agent
```

### Step 5: Security Configuration

#### 5.1 Create Service Account
```bash
gcloud iam service-accounts create trading-agent-sa \
    --display-name="Trading Agent Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:trading-agent-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudsql.client"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:trading-agent-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

#### 5.2 Store Secrets
```bash
# Store Zerodha API credentials
echo -n "your-zerodha-api-key" | gcloud secrets create zerodha-api-key --data-file=-
echo -n "your-zerodha-api-secret" | gcloud secrets create zerodha-api-secret --data-file=-
echo -n "your-zerodha-access-token" | gcloud secrets create zerodha-access-token --data-file=-

# Store OpenAI API key
echo -n "your-openai-api-key" | gcloud secrets create openai-api-key --data-file=-
```

### Step 6: Networking Configuration

#### 6.1 Create Firewall Rules
```bash
# Allow HTTP/HTTPS traffic
gcloud compute firewall-rules create allow-trading-dashboard \
    --allow tcp:80,tcp:443,tcp:5173 \
    --source-ranges 0.0.0.0/0 \
    --target-tags trading-agent

# Allow internal communication
gcloud compute firewall-rules create allow-internal-trading \
    --allow tcp:5432,tcp:6379 \
    --source-ranges 10.0.0.0/8 \
    --target-tags trading-agent
```

#### 6.2 Set up Load Balancer (Optional)
```bash
# Create instance group
gcloud compute instance-groups unmanaged create trading-agent-group \
    --zone=$ZONE

gcloud compute instance-groups unmanaged add-instances trading-agent-group \
    --instances=$INSTANCE_NAME \
    --zone=$ZONE

# Create health check
gcloud compute health-checks create http trading-agent-health-check \
    --port=5173 \
    --request-path=/

# Create backend service
gcloud compute backend-services create trading-agent-backend \
    --protocol=HTTP \
    --health-checks=trading-agent-health-check \
    --global

gcloud compute backend-services add-backend trading-agent-backend \
    --instance-group=trading-agent-group \
    --instance-group-zone=$ZONE \
    --global
```

### Step 7: Monitoring and Logging

#### 7.1 Set up Monitoring
```bash
# Create alerting policy for system health
gcloud alpha monitoring policies create --policy-from-file=monitoring-policy.yaml
```

Create `monitoring-policy.yaml`:
```yaml
displayName: "Trading Agent Health"
conditions:
  - displayName: "High CPU Usage"
    conditionThreshold:
      filter: 'resource.type="gce_instance" AND resource.label.instance_id="your-instance-id"'
      comparison: COMPARISON_GREATER_THAN
      thresholdValue: 0.8
      duration: 300s
notificationChannels:
  - "projects/your-project/notificationChannels/your-channel-id"
```

#### 7.2 Set up Log Aggregation
```bash
# Install logging agent
curl -sSO https://dl.google.com/cloudagents/add-logging-agent-repo.sh
sudo bash add-logging-agent-repo.sh
sudo apt-get update
sudo apt-get install google-fluentd
```

### Step 8: Application Deployment

#### 8.1 Deploy Application Code
```bash
# SSH into the instance
gcloud compute ssh $INSTANCE_NAME

# Navigate to application directory
cd /opt/trading-agent

# Pull latest code
git pull origin main

# Install/update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Run database migrations
python manage.py migrate

# Restart services
sudo supervisorctl restart trading-agent
```

#### 8.2 Deploy Dashboard
```bash
# Build React dashboard
cd trading-dashboard
npm install
npm run build

# Copy build files to nginx
sudo cp -r dist/* /var/www/html/

# Configure nginx
sudo cat > /etc/nginx/sites-available/trading-dashboard << EOF
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        root /var/www/html;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }
    
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/trading-dashboard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Configuration

### Environment Variables
Create a comprehensive `.env` file:
```bash
# Database Configuration
DATABASE_URL=postgresql://trading_user:password@localhost/trading_db
REDIS_URL=redis://localhost:6379

# Zerodha API Configuration
ZERODHA_API_KEY=your-api-key
ZERODHA_API_SECRET=your-api-secret
ZERODHA_ACCESS_TOKEN=your-access-token

# AI/ML Configuration
OPENAI_API_KEY=your-openai-key

# GCP Configuration
GCP_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Application Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_POSITION_SIZE=0.1
MAX_DAILY_LOSS=0.05
RISK_TOLERANCE=medium

# Monitoring Configuration
ENABLE_MONITORING=true
ALERT_EMAIL=your-email@domain.com
SLACK_WEBHOOK_URL=your-slack-webhook
```

### Risk Management Configuration
```python
# risk_config.py
RISK_LIMITS = {
    'max_position_size': 0.1,  # 10% per position
    'max_portfolio_exposure': 0.8,  # 80% total exposure
    'max_daily_loss': 0.05,  # 5% daily loss limit
    'max_drawdown': 0.15,  # 15% maximum drawdown
    'var_limit': 0.03,  # 3% VaR limit
    'max_correlation': 0.7,  # Maximum correlation
    'min_liquidity_ratio': 0.2,  # 20% cash minimum
    'max_leverage': 3.0,  # 3x maximum leverage
}
```

## Monitoring and Maintenance

### Health Checks
```bash
# System health check script
#!/bin/bash
curl -f http://localhost:5173/health || exit 1
python -c "import redis; r=redis.Redis(); r.ping()" || exit 1
python -c "import psycopg2; conn=psycopg2.connect('$DATABASE_URL'); conn.close()" || exit 1
```

### Log Monitoring
```bash
# Monitor application logs
tail -f /var/log/trading-agent.out.log

# Monitor error logs
tail -f /var/log/trading-agent.err.log

# Monitor system logs
journalctl -u trading-agent -f
```

### Performance Monitoring
```bash
# Monitor system resources
htop
iotop
nethogs

# Monitor database performance
psql $DATABASE_URL -c "SELECT * FROM pg_stat_activity;"

# Monitor Redis performance
redis-cli info
```

## Backup and Recovery

### Database Backup
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
gcloud sql export sql $DB_INSTANCE_NAME gs://$PROJECT_ID-backups/db_backup_$DATE.sql

# Local backup
pg_dump $DATABASE_URL > $BACKUP_DIR/local_backup_$DATE.sql

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "local_backup_*.sql" -mtime +7 -delete
```

### Model Backup
```bash
# Backup trained models
gsutil -m cp -r /opt/trading-agent/models gs://$PROJECT_ID-trading-models/backup_$DATE/
```

## Scaling and Optimization

### Auto Scaling
```bash
# Create instance template
gcloud compute instance-templates create trading-agent-template \
    --machine-type=n1-standard-4 \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --scopes=https://www.googleapis.com/auth/cloud-platform

# Create managed instance group
gcloud compute instance-groups managed create trading-agent-group \
    --template=trading-agent-template \
    --size=1 \
    --zone=$ZONE

# Set up auto scaling
gcloud compute instance-groups managed set-autoscaling trading-agent-group \
    --max-num-replicas=5 \
    --min-num-replicas=1 \
    --target-cpu-utilization=0.7 \
    --zone=$ZONE
```

### Performance Optimization
```python
# Optimize Python application
# Use connection pooling
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'OPTIONS': {
            'MAX_CONNS': 20,
            'CONN_MAX_AGE': 600,
        }
    }
}

# Use Redis for caching
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://localhost:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {'max_connections': 50}
        }
    }
}
```

## Security Best Practices

### 1. Network Security
- Use VPC with private subnets
- Implement firewall rules
- Use Cloud NAT for outbound traffic
- Enable DDoS protection

### 2. Data Security
- Encrypt data at rest and in transit
- Use Cloud KMS for key management
- Implement proper IAM policies
- Regular security audits

### 3. Application Security
- Use HTTPS everywhere
- Implement rate limiting
- Input validation and sanitization
- Regular dependency updates

### 4. API Security
- Secure API key storage
- Implement API rate limiting
- Use OAuth 2.0 where possible
- Monitor API usage

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues
```bash
# Check database connectivity
gcloud sql instances describe $DB_INSTANCE_NAME
psql $DATABASE_URL -c "SELECT 1;"
```

#### 2. Memory Issues
```bash
# Monitor memory usage
free -h
ps aux --sort=-%mem | head

# Optimize Python memory usage
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1
```

#### 3. API Rate Limiting
```python
# Implement exponential backoff
import time
import random

def api_call_with_backoff(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
    raise Exception("Max retries exceeded")
```

## Cost Optimization

### 1. Compute Optimization
- Use preemptible instances for non-critical workloads
- Implement auto-scaling
- Use committed use discounts
- Monitor and optimize resource usage

### 2. Storage Optimization
- Use appropriate storage classes
- Implement lifecycle policies
- Compress data where possible
- Regular cleanup of old data

### 3. Network Optimization
- Use regional resources
- Minimize data transfer
- Use CDN for static content
- Optimize API calls

## Support and Maintenance

### Regular Maintenance Tasks
1. **Daily**: Monitor system health, check logs
2. **Weekly**: Review performance metrics, update dependencies
3. **Monthly**: Security patches, backup verification
4. **Quarterly**: Performance optimization, cost review

### Emergency Procedures
1. **System Down**: Check health endpoints, restart services
2. **High CPU**: Scale up instances, optimize queries
3. **Memory Leak**: Restart application, investigate logs
4. **Database Issues**: Check connections, review slow queries

### Contact Information
- **Technical Support**: tech-support@yourcompany.com
- **Emergency Contact**: +1-XXX-XXX-XXXX
- **Documentation**: https://docs.yourcompany.com

---

This deployment guide provides a comprehensive framework for deploying the AI Trading Agent on Google Cloud Platform. Customize the configurations based on your specific requirements and security policies.

