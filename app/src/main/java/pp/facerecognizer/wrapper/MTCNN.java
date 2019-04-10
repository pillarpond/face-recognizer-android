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
import android.graphics.RectF;
import android.os.Trace;

import org.tensorflow.Graph;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import androidx.core.util.Pair;

public class MTCNN {
    private static final String MODEL_FILE = "file:///android_asset/mtcnn.pb";
    // Only return this many results.
    private static final int MAX_RESULTS = 100;
    private static final int BYTE_SIZE_OF_FLOAT = 4;

    // Config values.
    private String inputName;

    // Pre-allocated buffers.
    private FloatBuffer outputProbs;
    private FloatBuffer outputBoxes;
    private String[] outputNames;

    private TensorFlowInferenceInterface inferenceInterface;

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     */
    public static MTCNN create(
            final AssetManager assetManager) {
        final MTCNN d = new MTCNN();

        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);

        final Graph g = d.inferenceInterface.graph();

        d.inputName = "input";
        if (g.operation(d.inputName) == null)
            throw new RuntimeException("Failed to find input Node '" + d.inputName + "'");

        d.outputNames = new String[] {"prob", "landmarks", "box"};
        if (g.operation(d.outputNames[0]) == null)
            throw new RuntimeException("Failed to find output Node '" + d.outputNames[0] + "'");

        if (g.operation(d.outputNames[1]) == null)
            throw new RuntimeException("Failed to find output Node '" + d.outputNames[1] + "'");

        if (g.operation(d.outputNames[2]) == null)
            throw new RuntimeException("Failed to find output Node '" + d.outputNames[2] + "'");

        // Pre-allocate buffers.
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(MAX_RESULTS * BYTE_SIZE_OF_FLOAT);
        byteBuffer.order(ByteOrder.nativeOrder());
        d.outputProbs = byteBuffer.asFloatBuffer();

        d.outputBoxes = ByteBuffer.allocateDirect(MAX_RESULTS * BYTE_SIZE_OF_FLOAT * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();

        return d;
    }

    private MTCNN() {}

    public Pair[] detect(Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("detect");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        int w = bitmap.getWidth(), h = bitmap.getHeight();
        int intValues[] = new int[w * h];
        float floatValues[] = new float[w * h * 3];

        bitmap.getPixels(intValues, 0, w, 0, 0, w, h);

        // BGR
        for (int i = 0; i < intValues.length; ++i) {
            int p = intValues[i];

            floatValues[i * 3 + 0] = p & 0xFF;
            floatValues[i * 3 + 1] = (p >> 8) & 0xFF;
            floatValues[i * 3 + 2] = (p >> 16) & 0xFF;
        }
        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, h, w, 3);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, false);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputNames[0], outputProbs);
        inferenceInterface.fetch(outputNames[2], outputBoxes);
        Trace.endSection();

        outputProbs.flip();
        outputBoxes.flip();

        int len = outputProbs.remaining();
        Pair faces[] = new Pair[len];

        for (int i = 0; i < len; i++) {
            float top = outputBoxes.get();
            float left = outputBoxes.get();
            float bottom = outputBoxes.get();
            float right = outputBoxes.get();

            faces[i] = new Pair<>(
                    new RectF(left, top, right, bottom), outputProbs.get());
        }

        if (outputBoxes.hasRemaining())
            outputBoxes.position(outputBoxes.limit());

        outputProbs.compact();
        outputBoxes.compact();

        Trace.endSection(); // "detect"
        return faces;
    }

    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    public void close() {
        inferenceInterface.close();
    }
}
