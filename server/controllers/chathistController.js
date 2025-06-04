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
 * Stream functions handles requests for streaming,
 * It takes a URL and returns a function that handles the request.
 *
 * @param {*} url
 * @returns
 */
// const stream = (url, callback) => {
//   return async (req, res) => {
//     try {
//       const { prompt } = req.body;
//       let msg = '';
//       const response = await fetch(url, {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ prompt }),
//       });
//       const streamer = getStreamer(await response.body.getReader());
//       streamer.pipe(res);
//       streamer.on('complete', async (message) => {
//         msg += message;
//       });
//       streamer.on('end', async () => {
//         msg = msg.replace(/\n/g, ' ');
//         await callback(msg, req.body);
//       });
//     } catch (e) {
//       res.status(500).json({
//         status: 'error',
//         data: { message: 'Internal error!!!' },
//       });
//     }
//   };
// };

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
exports.getQuery = async (req, res) => {
  try {
    const { prompt, uuid } = req.body;
    const userConversation = await Conversation.create({
      uuid,
      role: 'user',
      content: prompt,
    });
    console.log('entering');
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

const generateRandomImages = (n, range, draft) => {
  let result = [];
  for (let i = 0; i < n; i++) {
    result.push(`${draft}/${Math.floor(Math.random() * range) + 1}.jpg`);
  }
  return result;
};

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

exports.getChats = async (req, res) => {
  try {
    const data = await Chat.find().sort({ createdAt: 1 });
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
