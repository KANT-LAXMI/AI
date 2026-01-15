# DOCKER 🐳

![alt text](image.png)

Docker is an open platform for developing , shipping and running applictaions.
OR It is a platform which packages an applictaion and all its dependencies together in the form of containers.

## Why we need docker ?

Developers often face issues where an app works on their computer but breaks elsewhere (e.g., production).

🧠 Why it happens: Different OS, Python/Java/C# versions, missing packages, etc.

## ✅ Docker solution:

Package everything (code + dependencies + OS environment) into a container that runs the same everywhere — on your laptop, in staging, or in the cloud.

## Docker file , Docker image and Docker conatiner

### 🚢 1. Dockerfile

A Dockerfile is a text file with instructions on how to build a Docker image i.e. text document which contains all command that a user can call on the command line to assemble/create an image

### 📦 2. Docker Image

A Docker Image is the result of building a Dockerfile. Template to create a docker container. It's a snapshot of an application with all its dependencies. It is the blueprint of creating docker container

### 🧱 3. Docker Container

A Docker Container is a running instance of a Docker Image.It hold entire package to run the applictaion. It’s isolated, lightweight, and can be started, stopped, or deleted.

Think of it like running the app from the image.

---

# Dockerizing our app

### ✅ Example: Simple Python App

#### 📄 Step 1: Dockerfile

Create a file named Dockerfile:

```
# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run the app
CMD ["python", "app.py"]
```

##### DOCKER FILE COMMAND

![alt text](image-56.png)

#### 🗃️ Step 2: Project Structure

```
demo/
├── app.py
├── requirements.txt
└── Dockerfile

```

app.py

```
print("Hello from Docker!")
```

requirements.txt

```
# no dependencies for this example
```

#### ⚙️ Step 3: Build the Docker Image

Move to the demo directory

```
Run this in the terminal:
docker build -t demo .
```

![alt text](image-2.png)

#### 🚀 Step 4: Run the Docker Container

```
docker run demo
```

![alt text](image-1.png)

# Docker command

-it stand for interactive mode i.e. we can acces the terminal of ruuning container (example: docker run -it ubuntu)

`1) version -> `
docker -v OR docker --version

`2) pull image from docker hub -> `
docker pull [image-name]

![alt text](image-8.png)
![alt text](image-17.png)

`3) run the image -> ` docker run [image-name]

`4) searches Docker Hub -> ` docker search mysql (It searches Docker Hub (the default public image registry) for images related to mysql.)
![alt text](image-3.png)

`5) lists all Docker containers on your system — both Running and Stopped -> ` docker ps -a
![alt text](image-4.png)

`6) rename my image -> ` docker run --name tester -d [Image name / container Id]
![alt text](image-5.png)

`7. to run container in interactive modec-> ` docker run --name tester(new image name) -it -d qwerty(image name )

![alt text](image-7.png)

`8) check only container which are running -> ` docker ps

`9) To enter in docker container -> ` docker exec -it [Container Id] [COMMAND]

![alt text](image-11.png)
![alt text](image-10.png)

`10) Inspect returns detailed, low-level information about a Docker container or image. -> ` docker inspect 9715289d1d92aef71592274307
![alt text](image-12.png)

`11) Remove a running container ->` docker rm [container-id]
![alt text](image-14.png)
![alt text](image-13.png)
docker rm -f [container-id]
![alt text](image-15.png)

`12) Start or stop a running image -> ` docker stop/restart [image-name]
![alt text](image-16.png)

`13) remove/destor image or container -> ` docker rmi Image_Name and
docker rm Container and one important point to note before deleting the image it respective container have to to removed first

`14) Port binding -> ` docker run -p <host_port>:<container_port> image_name
![alt text](image-19.png)
![alt text](image-18.png)

---

# How Docker works ?

![alt text](image-20.png)
The diagram below shows the architecture of Docker and how it works when we run “docker build”, “docker pull” and “docker run”.

There are 3 components in Docker architecture:

Docker client

The docker client talks to the Docker daemon.

### Docker host

The Docker daemon listens for Docker API requests and manages Docker objects such as images, containers, networks, and volumes.

### Docker registry

A Docker registry stores Docker images. Docker Hub is a public registry that anyone can use.

`Let’s take the “docker run” command as an example.`

Docker pulls the image from the registry.
Docker creates a new container.
Docker allocates a read-write filesystem to the container.
Docker creates a network interface to connect the container to the default network.
Docker starts the container.

---

# Difference between VM and Docker

![alt text](image-21.png)

---

# Docker Compose

It is a tool for defining and running multi-container applications.
When your app needs multiple containers (like a web server + database + cache). Docker Compose helps manage them all together.

For example:
`A Flask app / 
A PostgreSQL database / 
A Redis cache.
You can define all three in one docker-compose.yml`

Here's a simple docker-compose.yml example for a web app using Node.js and MongoDB, followed by a line-by-line explanation.

```
version: '3.8'

services:
  app:
    build: .
    ports:
      - "3000:3000"
    depends_on:
      - mongo
    environment:
      - MONGO_URL=mongodb://mongo:27017/mydb

  mongo:
    image: mongo:6
    volumes:
      - mongo-data:/data/db

volumes:
  mongo-data:

```

🧠 Explanation

### 🔹 version: '3.8'

Specifies the version of the Docker Compose file format.
Version 3.8 is commonly used and supports modern features.

### 🔹 services:

Defines the containers your app needs. In this case: app and mongo.

`1. app Service (Your Node.js App)`

```
    app:
        build:
```

Tells Docker to build an image for the app using the Dockerfile in the current directory.

```
    ports:
        - "3000:3000"
```

Maps host port 3000 to container port 3000, so you can access the app at localhost:3000.

```
    depends_on:
      - mongo
```

Ensures the mongo container starts before the app container.

```
    environment:
      - MONGO_URL=mongodb://mongo:27017/mydb
```

Sets an environment variable inside the container, typically used by the Node app to connect to MongoDB.

`2. mongo Service (MongoDB Database)`

```
    mongo:
        image: mongo:6
```

Uses the official MongoDB image (version 6) from Docker Hub.

```
    volumes:
      - mongo-data:/data/db
```

Mounts a named volume (mongo-data) to persist MongoDB data even if the container stops or is deleted.

`3. volumes:`

```
    volumes:
        mongo-data:
```

Declares a named volume to be used by the mongo service.
This ensures database data is stored outside the container’s filesystem.

`✅ Summary`

This setup runs:

A Node.js app that connects to...

A MongoDB database, using a volume to persist data.

We can run it all with:

```
docker-compose up
```

## Docker Compose Command

![alt text](image-22.png)
![alt text](image-23.png)
![alt text](image-24.png)
![alt text](image-25.png)
![alt text](image-26.png)
![alt text](image-27.png)

---

# 🌐 What is Docker Network?

Docker networking allows containers to communicate with

1. Each other without needing any kind of port or local host
2. The host system
3. The outside world (internet)

![alt text](image-64.png)
![alt text](image-65.png)
![alt text](image-66.png)
![alt text](image-67.png)
![alt text](image-68.png)

### Docker network command

`1) Lists all the Docker networks available on your system.` docker network ls

`2) Creates a new user-defined bridge network.`
docker network create Network_Name

![alt text](image-29.png)

![alt text](image-30.png)
![alt text](image-31.png)
Explanation:
`▶️ docker run -d`
docker run: Runs a new container.

-d: Detached mode – runs the container in the background.

`▶️ -p 27017:27017`
Maps port 27017 on your host to port 27017 inside the container.

MongoDB uses port 27017 by default, so this makes it accessible from your machine.

`▶️ --name mongo`
Names the container mongo so you can easily reference it (instead of using the container ID).

`▶️ --network mongo-network`
Connects the container to a user-defined Docker network called mongo-network.

This allows containers in the same network to communicate by name (like mongo, mongo-express, etc.).

⚠️ Make sure this network exists. If not, create it with:

docker network create mongo-network
`▶️ -e MONGO_INITDB_ROOT_USERNAME=admin`
Sets the root username for MongoDB to admin.

`▶️ -e MONGO_INITDB_ROOT_PASSWORD=qwerty`
Sets the root password for MongoDB to qwerty.

These credentials will be used to log into MongoDB securely.

`▶️ mongo`
Specifies the Docker image to use – in this case, the official MongoDB image from Docker Hub.

### Mongo express

It is the GUI for mongo db provided by docker hub.
![alt text](image-32.png)
![alt text](image-33.png)

we can acces the Mongo db gui by accesing localhost:8081

![alt text](image-34.png)

Below is the node js application is running on the port number 5050
![alt text](image-35.png)
after creating our own database name as apna college-db and having collections named as users
![alt text](image-37.png)
Onece the network connection is established we can able to acces the data from the mongodb which is running on port number 57017
![alt text](image-38.png)
![alt text](image-36.png)

---

# DOCKER COMPOSE

Docker Compose lets you define and run multi-container applications using a simple docker-compose.yml file.
![alt text](image-43.png)

![alt text](image-48.png)
![alt text](image-49.png)
![alt text](image-50.png)
![alt text](image-51.png)
![alt text](image-52.png)

## REASON

![alt text](image-40.png)
![alt text](image-41.png)

## DIFFERENCE ?

![alt text](image-39.png)

## WHAT TO USE

![alt text](image-42.png)

### example : -

##### docker-compose.yml

```
version: '3.8'  # Compose file version

services:
  app:  # Node.js application
    build: .  # Build image from Dockerfile in current directory
    container_name: node-app
    ports:
      - "3000:3000"  # Host:Container port mapping
    environment:
      - MONGO_URL=mongodb://admin:qwerty@mongo:27017  # Used by app to connect to MongoDB
    depends_on:
      - mongo  # Ensure mongo starts before app
    networks:
      - app-network

  mongo:  # MongoDB service
    image: mongo  # Official MongoDB image
    container_name: mongo
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=qwerty
    networks:
      - app-network

  mongo-express:  # Web-based MongoDB UI
    image: mongo-express
    container_name: mongo-express
    ports:
      - "8081:8081"
    environment:
      - ME_CONFIG_MONGODB_ADMINUSERNAME=admin
      - ME_CONFIG_MONGODB_ADMINPASSWORD=qwerty
      - ME_CONFIG_MONGODB_URL=mongodb://admin:qwerty@mongo:27017
    depends_on:
      - mongo
    networks:
      - app-network

# Network definition for inter-container communication
networks:
  app-network:
    driver: bridge


```

![alt text](image-45.png)
![alt text](image-44.png)

## Docker compose command

```
docker compose -f fileName.yml up -d
docker compose -f fileName.yml down
```

![alt text](image-46.png)
![alt text](image-47.png)

`NOTE : ` Now we are going to run the two container in the same network using the yml and docker compose.

1. `Created qwerty.yaml`

```
services:
  mongo:  # MongoDB service
    image: mongo
    container_name: mongo
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=qwerty
    networks:
      - app-network

  mongo-express:  # Web-based MongoDB UI
    image: mongo-express
    container_name: mongo-express
    ports:
      - "8081:8081"
    environment:
      - ME_CONFIG_MONGODB_ADMINUSERNAME=admin
      - ME_CONFIG_MONGODB_ADMINPASSWORD=qwerty
      - ME_CONFIG_MONGODB_URL=mongodb://admin:qwerty@mongo:27017
    depends_on:
      - mongo
    networks:
      - app-network

networks:
  app-network:
    driver: bridge

```

2. `Running the container `

![alt text](image-53.png)

We are able to connect the node application with our container mongodb database using docker compose

![alt text](image-54.png)

3. `Removing the running network , applictaion in the container`
   ![alt text](image-55.png)

---

# Docker volume

Volumes are persistent data store for containers
![alt text](image-57.png)
A volume in Docker is a special directory stored outside the container's filesystem, used to:

1. ✅ Persist data even when containers are deleted.

2. 🔄 Share data between multiple containers.

3. 📂 Keep the container lightweight and stateless.

![alt text](image-58.png)
![alt text](image-59.png)

### commands

1. NAMED VOLUMES

   ` docker run -v VOL_NAME:CONT_DIR`

   ![alt text](image-60.png)

2. ANONYMOUS VOLUMES

   `docker run -v MOUNT_PATH`

   ![alt text](image-61.png)

3. BIND MOUNT

   `docker run -v HOST_DIR:CONT_DIR`

   ![alt text](image-62.png)

4. DELETE UNUSED VOLUMES

   `docker volume prune`

### difference

![alt text](image-63.png)
