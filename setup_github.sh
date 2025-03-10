#!/bin/bash

# Initialize git repository
git init

# Add files
git add .
git commit -m "Initial commit"

# Set remote repository (replace with your repository URL)
echo "Enter your GitHub repository URL:"
read repo_url
git remote add origin $repo_url

# Push to GitHub
git push -u origin main
