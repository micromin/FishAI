package com.innoxlab.fishai;

import android.Manifest;
import android.app.Fragment;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Trace;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.UiThread;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Surface;
import android.view.View;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import com.innoxlab.fishai.tflite.BorderedText;
import com.innoxlab.fishai.tflite.CameraConnectionFragment;
import com.innoxlab.fishai.tflite.Classifier;
import com.innoxlab.fishai.tflite.ImageUtils;
import com.innoxlab.fishai.tflite.LegacyCameraConnectionFragment;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements ImageReader.OnImageAvailableListener,
        Camera.PreviewCallback{
    private String TAG = MainActivity.class.getSimpleName();

    private Fragment fragment;

    private static int RESULT_LOAD_IMAGE = 100;
    private static int RESULT_CAPTURE_IMAGE = 101;
    private static final int CAMERA_PERMISSION_CODE = 102;

    protected int previewWidth = 0;
    protected int previewHeight = 0;
    private HandlerThread handlerThread;
    private Handler handler;
    private Classifier classifier;
    private ListView listView;
    private static PredictionAdapter adapter;
    private TextView noPrediction;
    private FrameLayout frameLayout;

    private ArrayList<Classifier.Recognition> predictions;

    private Classifier.Model model = Classifier.Model.QUANTIZED;
    private Classifier.Device device = Classifier.Device.CPU;
    private int numThreads = 1;
    private boolean useCamera2API;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final float TEXT_SIZE_DIP = 10;
    private Bitmap rgbFrameBitmap = null;
    private long lastProcessingTimeMs;
    private Integer sensorOrientation;
    private BorderedText borderedText;
    private ImageView imageView;
    private boolean isProcessingFrame = false;
    private int[] rgbBytes = null;
    private byte[][] yuvBytes = new byte[3][];
    private int yRowStride;
    private Runnable imageConverter;
    private Runnable postInferenceCallback;

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
        imageView = findViewById(R.id.image_preview);
        frameLayout = findViewById(R.id.camera_container);
        noPrediction = findViewById(R.id.no_predictions);
        noPrediction.setVisibility(View.VISIBLE);
        predictions = new ArrayList<>();
        adapter = new PredictionAdapter(predictions, getApplicationContext());

        imageView.setVisibility(View.VISIBLE);
        listView.setAdapter(adapter);
        listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
            @Override
            public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                Classifier.Recognition prediction = predictions.get(position);
            }
        });

        Button buttonLoadImage = findViewById(R.id.load_button);
        Button buttonCaptureImage = findViewById(R.id.capture_button);
        Button buttonLive = findViewById(R.id.live_button);

        buttonLoadImage.setOnClickListener(l -> {
            predictions.clear();
            adapter.notifyDataSetChanged();
            if (fragment != null){
                getFragmentManager().beginTransaction().remove(fragment).commit();
            }
            imageView.setVisibility(View.VISIBLE);
            frameLayout.setVisibility(View.INVISIBLE);

            Intent i = new Intent(
                    Intent.ACTION_PICK,
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

            startActivityForResult(i, RESULT_LOAD_IMAGE);
        });

        buttonCaptureImage.setOnClickListener(l -> {
            predictions.clear();
            adapter.notifyDataSetChanged();
            if (fragment != null){
                getFragmentManager().beginTransaction().remove(fragment).commit();
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
                } else {
                    imageView.setVisibility(View.VISIBLE);
                    frameLayout.setVisibility(View.INVISIBLE);
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent, RESULT_CAPTURE_IMAGE);
                }
            }
        });

        buttonLive.setOnClickListener(l -> {
            isProcessingFrame = false;
            predictions.clear();
            adapter.notifyDataSetChanged();
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
                } else {
                    imageView.setVisibility(View.INVISIBLE);
                    frameLayout.setVisibility(View.VISIBLE);
                    setFragment();
                }
            }
        });

        runInBackground(() -> createClassifier(model, device, numThreads));
    }

    private String chooseCamera() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        try {
            for (final String cameraId : manager.getCameraIdList()) {
                final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

                final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
                    continue;
                }

                final StreamConfigurationMap map =
                        characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                if (map == null) {
                    continue;
                }

                useCamera2API =
                        (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                                || isHardwareLevelSupported(
                                characteristics, CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);
                Log.i(TAG, String.format("Camera API lv2?: %s", useCamera2API));
                return cameraId;
            }
        } catch (CameraAccessException e) {
            Log.e(TAG, "Not allowed to access camera", e);
        }

        return null;
    }

    protected void setFragment() {
        String cameraId = chooseCamera();
        if (useCamera2API) {
            CameraConnectionFragment camera2Fragment =
                    CameraConnectionFragment.newInstance(
                            new CameraConnectionFragment.ConnectionCallback() {
                                @Override
                                public void onPreviewSizeChosen(final Size size, final int rotation) {
                                    previewHeight = size.getHeight();
                                    previewWidth = size.getWidth();
                                    MainActivity.this.onPreviewSizeChosen(size, rotation);
                                }
                            },
                            this,
                            getLayoutId(),
                            getDesiredPreviewFrameSize());

            camera2Fragment.setCamera(cameraId);
            fragment = camera2Fragment;
        } else {
            fragment =
                    new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
        }

        getFragmentManager().beginTransaction().replace(R.id.camera_container, fragment).commit();
    }
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }


    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        createClassifier(model, device, 1);
        if (classifier == null) {
            Log.e(TAG, "No classifier on preview!");
            return;
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        Log.i(TAG, String.format("Camera orientation relative to screen canvas: %d", sensorOrientation));

        Log.i(TAG, String.format("Initializing at size %dx%d", previewWidth, previewHeight));
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);

    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    private boolean isHardwareLevelSupported(
            CameraCharacteristics characteristics, int requiredLevel) {
        int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
        if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
            return requiredLevel == deviceLevel;
        }
        // deviceLevel is not LEGACY, can use numerical sort
        return requiredLevel <= deviceLevel;
    }

    protected int getLayoutId() {
        return R.layout.camera_connection_fragment;
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

    protected int[] getRgbBytes() {
        imageConverter.run();
        return rgbBytes;
    }

    @Override
    public void onPreviewFrame(final byte[] bytes, final Camera camera) {
        if (isProcessingFrame) {
            Log.w(TAG, "Dropping frame!");
            return;
        }

        try {
            // Initialize the storage bitmaps once when the resolution is known.
            if (rgbBytes == null) {
                Camera.Size previewSize = camera.getParameters().getPreviewSize();
                previewHeight = previewSize.height;
                previewWidth = previewSize.width;
                rgbBytes = new int[previewWidth * previewHeight];
                onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
            }
        } catch (final Exception e) {
            Log.e(TAG, "Exception!", e);
            return;
        }

        isProcessingFrame = true;
        yuvBytes[0] = bytes;
        yRowStride = previewWidth;

        imageConverter =
                new Runnable() {
                    @Override
                    public void run() {
                        ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);
                    }
                };

        postInferenceCallback =
                new Runnable() {
                    @Override
                    public void run() {
                        camera.addCallbackBuffer(bytes);
                        isProcessingFrame = false;
                    }
                };


        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        processImage(rgbFrameBitmap, sensorOrientation);
    }

    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                Log.d(TAG, String.format("Initializing buffer %d at size %d", i, buffer.capacity()));
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    @Override
    public void onImageAvailable(final ImageReader reader) {
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            final Image image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
            Trace.beginSection("imageAvailable");
            final Image.Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            imageConverter =
                    new Runnable() {
                        @Override
                        public void run() {
                            ImageUtils.convertYUV420ToARGB8888(
                                    yuvBytes[0],
                                    yuvBytes[1],
                                    yuvBytes[2],
                                    previewWidth,
                                    previewHeight,
                                    yRowStride,
                                    uvRowStride,
                                    uvPixelStride,
                                    rgbBytes);
                        }
                    };

            postInferenceCallback =
                    new Runnable() {
                        @Override
                        public void run() {
                            image.close();
                            isProcessingFrame = false;
                        }
                    };
            rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
            processImage(rgbFrameBitmap, sensorOrientation);
        } catch (final Exception e) {
            Log.e(TAG, "Exception!", e);
            Trace.endSection();
            return;
        }
        Trace.endSection();
    }
}
