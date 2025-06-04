const { Readable } = require('stream');

/**
 * Returns a Readable stream from a given reader.
 * @param {*} reader
 * @returns Readable stream.
 */
const getStreamer = (reader) => {
  return new Readable({
    async read() {
      try {
        let message = '';
        const { done, value } = await reader.read();
        if (!done) {
          chunk = Buffer.from(value);
          this.push(chunk); // Convert Uint8Array to Buffer
          message += JSON.parse(chunk.toString())['response'];
          this.emit('complete', message);
        } else this.push(null); // Signal the end of the stream
      } catch (error) {
        this.destroy(error); // Handle any errors that occur during reading
      }
    },
  });
};

module.exports = getStreamer;
