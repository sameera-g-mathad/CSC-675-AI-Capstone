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
const stream = (url) => {
  return async (req, res) => {
    try {
      const { prompt } = req.body;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });
      const streamer = getStreamer(await response.body.getReader());
      streamer.pipe(res);
    } catch (e) {
      res.status(500).json({
        status: 'error',
        data: { message: 'Internal error!!!' },
      });
    }
  };
};

exports.getTitle = stream('http://127.0.0.1:8001/api/v1/title');
exports.getQuery = stream('http://127.0.0.1:8000/api/v1/query');
