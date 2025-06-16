FROM python:3.10-slim

# Install OS dependencies for Playwright
RUN apt-get update && apt-get install -y \
    wget gnupg curl unzip fontconfig locales \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libasound2 libxshmfence1 \
    libgtk-3-0 libx11-xcb1 libxss1 libxtst6 ca-certificates fonts-liberation \
    libappindicator3-1 lsb-release xdg-utils && \
    rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN python -m playwright install chromium

# Copy source code
COPY . .

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
