FROM python:3.8-slim
WORKDIR /app

# Install python packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Port for GRPC
EXPOSE 6000
# Port for REST
EXPOSE 9000

# Seldon runs as user 8888 thus change file permissions as such
RUN mkdir /.cache
RUN chown -R 8888 /.cache
RUN chown -R 8888 /app

# Define environment variables
ENV MODEL_NAME ReviewRatings
ENV PERSISTENCE 0
ENV SERVICE_TYPE MODEL

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE