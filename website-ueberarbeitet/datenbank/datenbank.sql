CREATE DATABASE datenbank;
USE datenbank;

CREATE TABLE user (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL
);

CREATE TABLE ml_models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    model_name VARCHAR(255) NOT NULL,
    model_data LONGBLOB NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user(id)
);

CREATE TABLE scalers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    scaler_name VARCHAR(255) NOT NULL,
    scaler_data LONGBLOB NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user(id)
);

CREATE TABLE nn_dropout_model (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    model_name VARCHAR(255),
    model_data LONGBLOB,
    FOREIGN KEY (user_id) REFERENCES user(id)
);

