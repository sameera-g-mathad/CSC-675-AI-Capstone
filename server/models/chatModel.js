const mongoose = require('mongoose');

/**
 * Schema for the Chat model.
 */
const ChatSchema = new mongoose.Schema({
  chat_title: {
    type: String,
    required: [true, 'A chat_title is required'],
    default: '',
  },
  uuid: {
    type: String,
    required: [true, 'A chat uuid is required'],
    unique: [true, 'A UUID must be unique'],
  },
  displayImage: {
    type: String,
    required: [true, 'A display image is a must'],
  },
  createdAt: {
    type: Date,
    default: new Date(),
  },
});

const Chat = mongoose.model('Chat', ChatSchema);

module.exports = Chat;
