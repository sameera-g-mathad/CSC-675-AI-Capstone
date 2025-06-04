const mongoose = require('mongoose');

const ConversationSchema = new mongoose.Schema({
  content: {
    type: String,
    required: [true, 'Content is required'],
  },
  uuid: {
    type: String,
    required: [true, 'A uuid is required'],
  },
  role: {
    type: String,
    required: true,
    enum: {
      values: ['user', 'bot'],
      message: 'Role must be either user or bot',
    },
  },
  createdAt: {
    type: Date,
    default: new Date(),
  },
});

const Conversation = mongoose.model('Conversation', ConversationSchema);

module.exports = Conversation;
