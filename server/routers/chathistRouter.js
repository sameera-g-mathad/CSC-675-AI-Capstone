const express = require('express');
const { getTitle, getQuery } = require('./../controllers/chathistController');

// Define the router
const router = express.Router();

// Define the routes for title and query

router.route('/title').post(getTitle);

router.route('/query').post(getQuery);

module.exports = router;
