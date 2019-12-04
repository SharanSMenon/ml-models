# Tensorflow.js Sentiment Analysis model

This neural network was built on tensorflow and exported to tensorflow.js format. It does sentiment analysis and it was trained on the IMDB dataset

## How to use

```js
// A Node.JS script
const tf = require("@tensorflow/tfjs");
const fetch = require('node-fetch');
function getSentiment(score) {
    if (score > 0.66) return 'Positive';
    if (score > 0.4) return 'Neutral';
    if (score < 0.4) return 'Negative';
}
function predict(text, model, metadata) {
    // eslint-disable-next-line no-useless-escape
    const trimmed = text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');
    const inputBuffer = tf.buffer([1, metadata.max_len], "float32");
    trimmed.forEach((word, i) => inputBuffer.set(metadata.word_index[word] + metadata.index_from, 0, i));
    const input = inputBuffer.toTensor();
    const predictOut = model.predict(input);
    let positivity = predictOut.dataSync()[0];
    predictOut.dispose();
    return positivity;
}
const getMetaData = async () => {
    const dt = await fetch("https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json")
    return dt.json()
}
async function run(text) {
    const url = "https://raw.githubusercontent.com/SharanSMenon/ml-models/master/sentiment_analysis/tfjs_model/model.json"
    const u2 = "https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json"
	  const model = await tf.loadLayersModel(u2);
    const metadata = await getMetaData();
    const prediction = predict(txt, model, metadata)
    console.log(getSentiment(prediction))

}
let text = "This is great"
run(text)
```

This is a node.js script and make sure that you have `@tensorflow/tfjs` and `node-fetch` installed.
