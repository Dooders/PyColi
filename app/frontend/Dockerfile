# Use the official Nginx image as the base
FROM nginx:alpine

# Copy the HTML, JavaScript, and CSS files to the Nginx directory
COPY . /usr/share/nginx/html

# CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]