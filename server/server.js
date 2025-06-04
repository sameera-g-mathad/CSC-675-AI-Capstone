const mongoose = require('mongoose');
const dotenv = require('dotenv');
const app = require('./app');

dotenv.config({ path: `${__dirname}/.env` });

mongoose
  .connect(process.env.MONGODB_URI)
  .then(() => console.log('DB connected'))
  .catch((err) => console.log('error connecting db'));

app.listen(4000, '0.0.0.0', () => console.log('server up!!'));
