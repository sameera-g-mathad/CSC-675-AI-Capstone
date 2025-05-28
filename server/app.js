const express = require('express');
const morgan = require('morgan');

// create app.
const app = express();

app.use(morgan('dev'));
// This middleware adds {body} in request
app.use(express.json());

app.use('/health', (req, res) => {
  res.status(200).json({
    status: 'success',
    data: {
      message: 'healthy',
    },
  });
});

module.exports = app;
