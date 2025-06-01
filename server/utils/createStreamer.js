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
        const { done, value } = await reader.read();
        if (!done)
          this.push(Buffer.from(value)); // Convert Uint8Array to Buffer
        else this.push(null); // Signal the end of the stream
      } catch (error) {
        this.destroy(error); // Handle any errors that occur during reading
      }
    },
  });
};

module.exports = getStreamer;
