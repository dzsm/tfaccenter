import {Component, OnInit} from '@angular/core';
import * as tf from '@tensorflow/tfjs';

const DEFAULT_CHAR = ' ';
const CHARS_LOWER = 'abcdefghijklmnopqrstuvwxyzáéíóöúüőű';
const CHARS_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÖÚÜŐŰ';
const CHARS_LOWER_NO_ACCENT = 'abcdefghijklmnopqrstuvwxyzaeioouuou';

const INTEREST_SET = 'aeiouAEIOUáéíóöúüőűÁÉÍÓÖÚÜŐŰ';

@Component({
    selector: 'app-root',
    templateUrl: './app.component.html',
    styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {

    linearModel: tf.Sequential;

    trainingText = 'Az írott szöveg az emberiség történelmében hatalmas előrelépés, hiszen így a történelme folyamán egyedüli módon lehetővé vált az információ személytől térben és időben független tárolása szemben a szájhagyománnyal, amely mind térben mind időben adott személyhez vagy személyekhez kötött. A történelemről ránk maradt információk legnagyobb része a XX. századig írásos szövegemlékekből áll. Azok a szövegek, amelyek olyan kultúráktól származnak, ahol az írásos információrögzítés létezik, a szövegek felépítése alapvetően különbözik az olyan kultúrák szövegeitől, ahol információk csak szájhagyomány útján maradtak fenn. A társadalomtudományokban a szöveges hagyomány nélküli kultúrákat nagyrészt az ókori ill. történelme előtti kultúrákhoz sorolják. Így a társadalomtudományban létezik a kultúrának egy olyan fontos meghatározása, amelynek alapjául közvetetten bár de a szöveg szolgál.';
    predictingText = 'Így a társadalomtudományban létezik a kultúrának';
    prediction: any;

    isWorking = false;

    windowSize = 5;

    margin: number;

    onehotMap: any;
    inverseOnehotMap: any;

    onehot4Map: any;

    accentMap: any;
    inverseAccentMap: any;

    ngOnInit() {
        this.createModel();
    }

    createModel() {

        this.margin = Math.round((this.windowSize - 1) / 2);

        const charset = Array.from(new Set(Array.from(CHARS_LOWER_NO_ACCENT + DEFAULT_CHAR)));

        this.onehotMap = new Map(charset.map(
            (c, i) => [c, Array(charset.length).fill(0).map((_, j) => (i === j ? 1 : 0))] as [string, any]
        ));

        this.inverseOnehotMap = new Map(charset.map(
            (c, i) => [i, c] as [number, string]
        ));

        this.onehot4Map = new Map(Array(4).fill(null).map(
            (v, i) => [i, Array(4).fill(0).map((_, j) => (i === j ? 1 : 0))] as [number, any]
        ));

        this.accentMap = new Map(Object.entries({
            'á': ['a', 1], 'é': ['e', 1], 'í': ['i', 1], 'ó': ['o', 1], 'ö': ['o', 2],
            'ő': ['o', 3], 'ú': ['u', 1], 'ü': ['u', 2], 'ű': ['u', 3]
        }));

        this.inverseAccentMap = new Map();

        this.accentMap.forEach((v, k) => {
            this.inverseAccentMap.set(v[0], new Map());
        });
        this.accentMap.forEach((v, k) => {
            this.inverseAccentMap.get(v[0]).set(v[1], k);
        });

        this.linearModel = tf.sequential();
        this.linearModel.add(tf.layers.dense({units: 40, inputShape: [charset.length * this.windowSize], activation: 'relu'}));
        this.linearModel.add(tf.layers.dense({units: 4, activation: 'sigmoid'}));

        this.linearModel.compile({loss: tf.losses.softmaxCrossEntropy, optimizer: 'adam', metrics: ['accuracy']});

    }

    async train(originalText): Promise<any> {

        this.isWorking = true;

        const dataSet = this.prepare(originalText);
        const xs = tf.tensor2d(dataSet.xt);
        const ys = tf.tensor2d(dataSet.yt);

        await this.linearModel.fit(xs, ys, {epochs: 100});

        this.isWorking = false;

    }

    predict(originalText) {

        this.isWorking = true;

        const dataSet = this.prepare(originalText);
        const xs = tf.tensor2d(dataSet.xt);

        const prediction = this.linearModel.predict(xs) as any;
        const output = tf.argMax(prediction, 1);

        const ys = Array.from(output.dataSync());

        const textArray = originalText.split('');

        for (let i = 0; i < dataSet.centers.length; i++) {

            const ci = dataSet.centers[i];
            const ma = this.inverseAccentMap.get(ci[0]);

            const ind = dataSet.indexes[i];
            textArray[ind - this.margin] = ma.get(ys[i]) || ci[0];

        }

        this.prediction = textArray.join('');

        this.isWorking = false;

    }


    _argMax(array) {
        return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
    }


    filter(c) {

        if (CHARS_LOWER.indexOf(c) < 0) {
            return ' ';
        }
        return c;

    }

    prepare(originalText) {

        const text = ' '.repeat(this.margin) + originalText + ' '.repeat(this.margin);

        const windows = [];
        const centers = [];
        const indexes = [];

        for (let i = 0; i < text.length; i++) {

            const c = this.filter(text.charAt(i));

            if (INTEREST_SET.indexOf(c) > -1) {
                const w = text.substr(i - this.margin, this.windowSize);
                windows.push(Array.from(w).map(a => {
                    a = this.filter(a);
                    return this.accentMap.get(a) ? this.accentMap.get(a)[0] : a;
                }));
                centers.push(this.accentMap.get(c) || [c, 0]);
                indexes.push(i);
            }
        }

        const xt = windows.map(w => w.reduce((a, n) => a.concat(this.onehotMap.get(n)), []));
        const yt = centers.map(y => this.onehot4Map.get(y[1]));

        return {windows, centers, indexes, xt, yt};

    }


    async loadModel() {
        //this.linearModel = await tf.loadModel('/assets/model.json');
    }

}
