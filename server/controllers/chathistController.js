const Chat = require('./../models/chatModel');
const Conversation = require('./../models/ConversationModel');

const first_draft_range = 21;
const second_draft_range = 34;
const third_draft_range = 31;
/**
 * Controller for handling chat history requests.
 * It fetches data from a specified URL and streams the response.
 */
// get the steram from utils.
const getStreamer = require('./../utils/createStreamer');

/**
 * Controller function to get the title for a chat based on a prompt.
 * It sends a POST request to a local server to generate the title.
 */
exports.getTitle = async (req, res) => {
  try {
    const { uuid, prompt } = req.body;
    let msg = '';
    const response = await fetch('http://127.0.0.1:8001/api/v1/title', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt }),
    });
    const streamer = getStreamer(await response.body.getReader());
    streamer.pipe(res);
    streamer.on('complete', async (message) => {
      msg += message;
    });
    streamer.on('end', async () => {
      msg = msg.replace(/\n/g, ' ');
      const data = await Chat.updateOne(
        { uuid },
        { chat_title: msg },
        { runValidators: true }
      );
    });
  } catch (e) {
    res.status(500).json({
      status: 'error',
      data: { message: 'Internal error!!!' },
    });
  }
};

/**
 * Controller function to handle a query request.
 * It creates a user conversation, sends a POST request to a local server,
 * and streams the response back to the client.
 * It also creates a bot conversation with the response message.
 */
exports.getQuery = async (req, res) => {
  try {
    const { prompt, uuid } = req.body;
    const userConversation = await Conversation.create({
      uuid,
      role: 'user',
      content: prompt,
    });
    let msg = '';
    const response = await fetch('http://127.0.0.1:8000/api/v1/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt }),
    });
    const streamer = getStreamer(await response.body.getReader());
    streamer.pipe(res);
    streamer.on('complete', async (message) => {
      msg += message;
    });
    streamer.on('end', async () => {
      msg = msg.replace(/\n/g, ' ');
      const llmResponse = await Conversation.create({
        uuid,
        role: 'bot',
        content: msg,
      });
    });
  } catch (e) {
    console.log(e);
    res.status(500).json({
      status: 'error',
      data: { message: 'Internal error!!!' },
    });
  }
};

/**
 * Generates an array of random image URLs based on the specified number,
 * range, and draft type.
 */
const generateRandomImages = (n, range, draft) => {
  let result = [];
  for (let i = 0; i < n; i++) {
    result.push(`${draft}/${Math.floor(Math.random() * range) + 1}.jpg`);
  }
  return result;
};

/**
 * Generates a random image URL based on a random draft type.
 * It randomly selects a draft type (first, second, or third) and generates
 * a random image URL within the specified range for that draft type.
 */
const generateRandomImage = () => {
  const draftNumber = Math.floor(Math.random() * 3) + 1;
  let range = 0;
  if (draftNumber === 1) {
    range = first_draft_range;
    draft = 'first_draft';
  } else if (draftNumber === 2) {
    range = second_draft_range;
    draft = 'second_draft';
  } else {
    range = third_draft_range;
    draft = 'third_draft';
  }
  return `${draft}/${Math.floor(Math.random() * range) + 1}.jpg`;
};

/**
 * Controller function to explore images.
 * It randomly selects a draft type and generates an array of random image URLs
 * based on that draft type. The generated images are then sent back in the response.
 * If the draft type is not recognized, it defaults to generating images from the second draft.
 */
exports.exploreImages = (req, res) => {
  let result = [];
  const draftNumber = Math.floor(Math.random() * 3) + 1;
  switch (draftNumber) {
    case 1:
      result = generateRandomImages(
        (n = 10),
        (range = first_draft_range),
        (draft = 'first_draft')
      );
      break;
    case 2:
      result = generateRandomImages(
        (n = 10),
        (range = second_draft_range),
        (draft = 'second_draft')
      );
      break;
    case 3:
      result = generateRandomImages(
        (n = 10),
        (range = third_draft_range),
        (draft = 'third_draft')
      );
      break;
    default:
      result = generateRandomNumbers(
        (n = 10),
        (range = second_draft_range),
        (draft = 'second_draft')
      );
      break;
  }
  res.status(200).json({
    status: 'success',
    data: {
      result,
    },
  });
};

/**
 * Controller function to create a new chat.
 * It generates a random image for the chat and creates a new chat entry in the database.
 * The chat is created with a default title and the provided UUID.
 * If the chat creation is successful, it returns the created chat data in the response.
 * If there is an error during chat creation, it returns a failure response with the error details
 */
exports.createChat = async (req, res) => {
  try {
    const { uuid } = req.body;
    const randomImage = generateRandomImage();
    const data = await Chat.create({
      chat_title: 'New Title',
      uuid,
      displayImage: randomImage,
    });
    res.status(200).json({
      status: 'success',
      data,
    });
  } catch (error) {
    res.status(404).json({
      status: 'failure',
      data: {
        error,
      },
    });
  }
};

/**
 * Controller function to get all chats.
 * It retrieves all chat entries from the database, sorted by creation date in descending order.
 * If the retrieval is successful, it returns the chat data in the response.
 * If there is an error during retrieval, it returns a failure response with the error details.
 */
exports.getChats = async (req, res) => {
  try {
    const data = await Chat.find().sort({ createdAt: -1 });
    res.status(200).json({
      status: 'success',
      data,
    });
  } catch (error) {
    res.status(404).json({
      status: 'failure',
      data: {
        error,
      },
    });
  }
};

/**
 * Controller function to get messages for a specific chat.
 * It retrieves the chat details and all conversations associated with the specified UUID.
 * The chat details include the chat title and display image.
 * The conversations are sorted by creation date in ascending order.
 * If the retrieval is successful, it returns the chat and conversations data in the response.
 * If there is an error during retrieval, it returns a failure response with the error details.
 */
exports.getMessages = async (req, res) => {
  try {
    const { uuid } = req.body;
    const chat = await Chat.find({ uuid }).select({
      chat_title: 1,
      displayImage: 1,
    });
    const conversations = await Conversation.find({ uuid })
      .sort({ createdAt: 1 })
      .select({ content: 1, role: 1 });
    res.status(200).json({
      status: 'success',
      data: {
        ...chat,
        conversations,
      },
    });
  } catch (error) {
    res.status(404).json({
      status: 'failure',
      data: {
        error,
      },
    });
  }
};
