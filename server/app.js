const cors = require('cors');
const express = require('express');
const morgan = require('morgan');
const chathistRouter = require('./routers/chathistRouter');

// create app.
const app = express();

app.use(cors());

app.use(morgan('dev'));

// This middleware adds {body} in request
app.use(express.json());

app.use('/api/v1', chathistRouter);

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
