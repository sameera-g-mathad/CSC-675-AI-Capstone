const cors = require('cors');
const express = require('express');
const morgan = require('morgan');
const chathistRouter = require('./routers/chathistRouter');

// create app.
const app = express();

app.use(cors());

// This middleware logs the request details in the console.
app.use(morgan('dev'));

// This middleware adds {body} in request
app.use(express.json());

app.use('/api/v1', chathistRouter);

// This middleware handles all the requests that are not defined in the routes.
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
