const express = require('express');
const morgan = require('morgan');
const chathistRouter = require('./routers/chathistRouter');

// create app.
const app = express();

app.use(morgan('dev'));
// This middleware adds {body} in request
app.use(express.json());

app.use('/api/v1', chathistRouter);

// app.post('/api/v1/title', async (req, res) => {
//   try {
//     const { prompt } = req.body;
//     const response = await fetch('http://127.0.0.1:8001/api/v1/title', {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ prompt }),
//     });
//     const data = await response.json();
//     res.status(200).json(data);
//   } catch (e) {
//     res.status(500).json({
//       status: 'error',
//       data: { message: 'Internal error!!!' },
//     });
//   }
// });

// app.post('/api/v1/query', async (req, res) => {
//   try {
//     const { prompt } = req.body;
//     const response = await fetch('http://127.0.0.1:8000/api/v1/query', {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ prompt }),
//     });
//     const reader = await response.body.getReader();
//     const streamer = new Readable({
//       async read() {
//         try {
//           const { done, value } = await reader.read();
//           if (!done) {
//             this.push(Buffer.from(value));
//           } else this.push(null);
//         } catch (err) {
//           this.destroy(err);
//         }
//       },
//     });
//     streamer.pipe(res);
//   } catch (error) {
//     console.log(error);
//     res.status(500).json({
//       status: 'error',
//       data: { message: 'Internal error!!!' },
//     });
//   }
// });

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
