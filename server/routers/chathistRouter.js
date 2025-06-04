const express = require('express');
const {
  createChat,
  getTitle,
  getQuery,
  getChats,
  exploreImages,
  getMessages,
} = require('./../controllers/chathistController');

// Define the router
const router = express.Router();

// Define the routes for title and query

router.route('/title').post(getTitle);

router.route('/query').post(getQuery);

router.route('/explore').get(exploreImages);

router.route('/createChat').post(createChat);

router.route('/getChats').get(getChats);

router.route('/getMessages').post(getMessages);

module.exports = router;
