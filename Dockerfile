# Dockerfile
#
# WHAT THIS DOES
# --------------
# Packages the entire diabetic optimizer into one Docker container:
#   1. Installs Python + Node.js
#   2. Installs all Python dependencies
#   3. Builds the React frontend (outputs to frontend/dist/)
#   4. Trains both ML models (LSTM + RL agent)
#   5. Runs the FastAPI server which also serves the built frontend
#
# Build stages:
#   Stage 1 (frontend-builder): builds React → produces frontend/dist/
#   Stage 2 (final):            Python app + copies frontend/dist/ in
#
# WHY TWO STAGES?
# ---------------
# Node.js and npm are only needed to BUILD the frontend (run npm build).
# They're not needed at runtime — the build output is just HTML/CSS/JS.
# Using a multi-stage build keeps the final image small by leaving
# Node.js behind.

# ─────────────────────────────────────────────
# Stage 1: Build the React frontend
# ─────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Copy package files first (Docker cache layer — only re-installs
# npm packages if package.json changes, not on every code change)
COPY frontend/package.json frontend/package-lock.json* ./

RUN npm install

# Copy the rest of the frontend source and build it
COPY frontend/ ./
RUN npm run build
# Output: /app/frontend/dist/ (static HTML/CSS/JS, ready to serve)


# ─────────────────────────────────────────────
# Stage 2: Python application
# ─────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# (requirements.txt copied first for Docker cache efficiency)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Copy the built React frontend from Stage 1 into our Python image
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create the models/saved directory (models will be trained here)
RUN mkdir -p models/saved

# Train both ML models during the build
# This means the container starts up instantly with pre-trained models
RUN python train_all.py

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Start the FastAPI server
# - host 0.0.0.0 means "accept connections from outside the container"
# - port 8000 matches EXPOSE above
# - no --reload in production (that's for development only)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]