import {adjustedScores} from './consumer/node_modules/ubukit-js/src/external-metrics.js';
self.onmessage = event => self.postMessage(adjustedScores(event.data.a,event.data.b));
