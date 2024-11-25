CREATE DATABASE parking_db;

USE parking_db;

CREATE TABLE parking_spaces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    location VARCHAR(255),
    status ENUM('available', 'occupied') DEFAULT 'available'
);

-- Add some sample parking spaces
INSERT INTO parking_spaces (location, status) VALUES ('A1', 'available');
INSERT INTO parking_spaces (location, status) VALUES ('A2', 'available');
