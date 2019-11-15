package com.innoxlab.fishai;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.UiThread;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import com.innoxlab.fishai.tflite.Classifier;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private String TAG = MainActivity.class.getSimpleName();

    private static int RESULT_LOAD_IMAGE = 100;
    private static int RESULT_CAPTURE_IMAGE = 101;
    private static final int CAMERA_PERMISSION_CODE = 102;

    private HandlerThread handlerThread;
    private Handler handler;
    private Classifier classifier;
    private ListView listView;
    private static PredictionAdapter adapter;
    private TextView noPrediction;

    private ArrayList<Classifier.Recognition> predictions;

    private Classifier.Model model = Classifier.Model.QUANTIZED;
    private Classifier.Device device = Classifier.Device.CPU;
    private int numThreads = 1;

    @Override
    public synchronized void onResume() {
        Log.d(TAG, "onResume " + this);
        super.onResume();

        if (handler == null) {
            initHandler();
        }
    }

    private void initHandler() {
        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    @Override
    public synchronized void onPause() {
        Log.d(TAG, "onPause " + this);

        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (final InterruptedException e) {
            Log.e(TAG, "Exception!", e);
        }

        super.onPause();
    }

    @Override
    public synchronized void onStop() {
        Log.d(TAG, "onStop " + this);
        super.onStop();
    }

    @Override
    public synchronized void onDestroy() {
        Log.d(TAG, "onDestroy " + this);
        super.onDestroy();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        setTitle("FishAI");
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        listView = findViewById(R.id.result_items);
        noPrediction = findViewById(R.id.no_predictions);
        noPrediction.setVisibility(View.VISIBLE);
        predictions = new ArrayList<>();
        adapter = new PredictionAdapter(predictions, getApplicationContext());

        listView.setAdapter(adapter);
        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Classifier.Recognition prediction = predictions.get(position);
            }
        });

        Button buttonLoadImage = findViewById(R.id.load_button);
        Button buttonCaptureImage = findViewById(R.id.capture_button);
        buttonLoadImage.setOnClickListener(l -> {
            Intent i = new Intent(
                    Intent.ACTION_PICK,
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

            startActivityForResult(i, RESULT_LOAD_IMAGE);
        });

        buttonCaptureImage.setOnClickListener(l -> {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
                } else {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, RESULT_CAPTURE_IMAGE);
                }
            }
        });

        runInBackground(() -> createClassifier(model, device, numThreads));
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();
                Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, RESULT_CAPTURE_IMAGE);
            } else {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler == null) {
            initHandler();
        }

        Log.d(TAG, "not null!");
        handler.post(r);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();

            InputStream imageStream = null;
            try {
                imageStream = getContentResolver().openInputStream(selectedImage);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
            Bitmap selected = BitmapFactory.decodeStream(imageStream);

            ImageView imageView = findViewById(R.id.image_preview);
            imageView.setImageBitmap(selected);

            Log.d(TAG, "processing image!");
            runInBackground(() -> processImage(selected, 90));
        } else if (requestCode == RESULT_CAPTURE_IMAGE && resultCode == RESULT_OK) {
            ImageView imageView = findViewById(R.id.image_preview);
            Bitmap photo = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(photo);

            Log.d(TAG, "processing image!");
            runInBackground(() -> processImage(photo, 90));
        }
    }

    private void createClassifier(Classifier.Model model, Classifier.Device device, int numThreads) {
        if (classifier != null) {
            Log.d(TAG, "Closing classifier.");
            classifier.close();
            classifier = null;
        }

        if (device == Classifier.Device.GPU && model == Classifier.Model.QUANTIZED) {
            Log.d(TAG, "Not creating classifier: GPU doesn't support quantized models.");
            runOnUiThread(
                    () -> {
                        Toast.makeText(this, "GPU does not yet supported quantized models.", Toast.LENGTH_LONG)
                                .show();
                    });
            return;
        }
        try {
            Log.d(TAG, String.format("Creating classifier (model=%s, device=%s, numThreads=%d)", model, device, numThreads));
            classifier = Classifier.create(this, model, device, numThreads);
        } catch (IOException e) {
            Log.e(TAG, "Failed to create classifier.", e);
        }
    }

    private void processImage(final Bitmap bitmap, int sensorOrientation) {
        Log.d(TAG, "Heree!!!");
        final List<Classifier.Recognition> results =
                classifier.recognizeImage(bitmap, sensorOrientation);
        Log.v(TAG, String.format("Detect: %s", results));
        runOnUiThread(
                () -> showResultsInUI(results));
    }

    @UiThread
    private void showResultsInUI(List<Classifier.Recognition> results) {
        predictions.clear();

        int counter = 0;
        while (results != null && counter < results.size() && counter < 3) {
            Classifier.Recognition recognition = results.get(counter);
            if (recognition != null) {
                predictions.add(recognition);
            }
            counter++;
        }

        if (predictions.size() == 0) {
            noPrediction.setVisibility(View.VISIBLE);
        } else {
            noPrediction.setVisibility(View.INVISIBLE);
        }

        adapter.notifyDataSetChanged();
    }
}
