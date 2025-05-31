const express = require('express');
const morgan = require('morgan');

// create app.
const app = express();

app.use(morgan('dev'));
// This middleware adds {body} in request
app.use(express.json());

app.post('/api/v1/title', async (req, res) => {
  try {
    const { prompt } = req.body;
    const response = await fetch('http://chat_title:8001/api/v1/title', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt }),
    });
    const data = await response.json();
    res.status(200).json(data);
  } catch (e) {
    res.status(500).json({
      status: 'error',
      data: { message: 'Internal error!!!' },
    });
  }
});

app.use('/health', (req, res) => {
  res.status(200).json({
    status: 'success',
    data: {
      message: 'healthy',
    },
  });
});

app.use('/*all', (req, res) => {
  console.log(req.hostname);
  res.status(404).json({
    status: 'error',
    data: {
      message: `Sorry http://${req.hostname}${req.originalUrl} doesn't exist.`,
    },
  });
});
module.exports = app;
