# Use the official Nginx image as the base
FROM nginx:alpine

# Copy the HTML, JavaScript, and CSS files to the Nginx directory
COPY . /usr/share/nginx/html

# Copy a custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Use environment variable in JavaScript and print the content
CMD ["nginx", "-g", "daemon off;"]