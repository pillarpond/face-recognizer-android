/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package pp.facerecognizer.wrapper;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Rect;
import android.os.Trace;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import pp.facerecognizer.Classifier;

public class FaceNet {
    private static final String MODEL_FILE = "file:///android_asset/facenet.pb";
    private static final int BYTE_SIZE_OF_FLOAT = 4;

    // Config values.
    private String inputName;
    private int inputHeight;
    private int inputWidth;

    // Pre-allocated buffers.
    private int[] intValues;
    private short[] shortValues;
    private FloatBuffer inputBuffer;

    private FloatBuffer outputBuffer;
    private String[] outputNames;

    private TensorFlowInferenceInterface inferenceInterface;

    private Bitmap bitmap;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     */
    public static FaceNet create(
            final AssetManager assetManager,
            final int inputHeight,
            final int inputWidth) {
        final FaceNet d = new FaceNet();

        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);

        final Graph g = d.inferenceInterface.graph();

        d.inputName = "input";
        // The inputName node has a shape of [N, H, W, C], where
        // N is the batch size
        // H = W are the height and width
        // C is the number of channels (3 for our purposes - RGB)
        final Operation
                inputOp1 = g.operation(d.inputName);
        if (inputOp1 == null) {
            throw new RuntimeException("Failed to find input Node '" + d.inputName + "'");
        }

        d.inputHeight = inputHeight;
        d.inputWidth = inputWidth;

        d.outputNames = new String[] {"embeddings"};
        final Operation outputOp1 = g.operation(d.outputNames[0]);
        if (outputOp1 == null) {
            throw new RuntimeException("Failed to find output Node'" + d.outputNames[0] + "'");
        }

        // Pre-allocate buffers.
        d.intValues = new int[inputHeight * inputWidth];
        d.shortValues = new short[inputHeight * inputWidth * 3];
        d.inputBuffer = ByteBuffer.allocateDirect(inputHeight * inputWidth * BYTE_SIZE_OF_FLOAT * 3)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();

        d.outputBuffer = ByteBuffer.allocateDirect(Classifier.EMBEDDING_SIZE * BYTE_SIZE_OF_FLOAT)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();

        d.bitmap = Bitmap.createBitmap(inputWidth, inputHeight, Config.ARGB_8888);
        return d;
    }

    private FaceNet() {}

    public FloatBuffer getEmbeddings(Bitmap originalBitmap, Rect rect) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("getEmbeddings");

        Trace.beginSection("preprocessBitmap");
        Canvas canvas = new Canvas(bitmap);
        canvas.drawBitmap(originalBitmap, rect, new Rect(0, 0, inputWidth, inputHeight), null);

        bitmap.getPixels(intValues, 0, inputWidth, 0, 0, inputWidth, inputHeight);

        for (int i = 0; i < intValues.length; ++i) {
            int p = intValues[i];

            shortValues[i * 3 + 2] = (short) (p & 0xFF);
            shortValues[i * 3 + 1] = (short) ((p >> 8) & 0xFF);
            shortValues[i * 3 + 0] = (short) ((p >> 16) & 0xFF);
        }

        double sum = 0f;
        for (short shortValue : shortValues) {
            sum += shortValue;
        }
        double mean = sum / shortValues.length;
        sum = 0f;

        for (short shortValue : shortValues) {
            sum += Math.pow(shortValue - mean, 2);
        }
        double std = Math.sqrt(sum / shortValues.length);
        double std_adj = Math.max(std, 1.0/Math.sqrt(shortValues.length));

        inputBuffer.rewind();
        for (short shortValue : shortValues) {
            inputBuffer.put((float) ((shortValue - mean) * (1 / std_adj)));
        }
        inputBuffer.flip();

        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, inputBuffer, 1, inputHeight, inputWidth, 3);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, false);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        outputBuffer.rewind();
        inferenceInterface.fetch(outputNames[0], outputBuffer);
        outputBuffer.flip();
        Trace.endSection();

        Trace.endSection(); // "getEmbeddings"
        return outputBuffer;
    }

    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    public void close() {
        inferenceInterface.close();
    }
}
