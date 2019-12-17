package com.innoxlab.objectai;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import com.innoxlab.objectai.tflite.Classifier;

import java.util.ArrayList;
import java.util.Locale;

public class PredictionAdapter extends ArrayAdapter<Classifier.Recognition> implements View.OnClickListener {

    private ArrayList<Classifier.Recognition> dataSet;
    private Context mContext;

    private static class ViewHolder {
        TextView detectedItem;
        TextView detectionValue;
    }

    public PredictionAdapter(ArrayList<Classifier.Recognition> data, Context context) {
        super(context, R.layout.prediction_item, data);
        this.dataSet = data;
        this.mContext = context;
    }

    @Override
    public void onClick(View v) {
        int position = (Integer) v.getTag();
        Classifier.Recognition clicked = getItem(position);
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        Classifier.Recognition model = getItem(position);
        ViewHolder viewHolder;

        if (convertView == null) {

            viewHolder = new ViewHolder();
            LayoutInflater inflater = LayoutInflater.from(getContext());
            convertView = inflater.inflate(R.layout.prediction_item, parent, false);
            viewHolder.detectedItem = convertView.findViewById(R.id.detected_item);
            viewHolder.detectionValue = convertView.findViewById(R.id.detected_item_value);
            convertView.setTag(viewHolder);
        } else {
            viewHolder = (ViewHolder) convertView.getTag();
        }

        if (model != null) {
            if (model.getTitle() != null) {
                viewHolder.detectedItem.setText(model.getTitle());
            }
            if (model.getConfidence() != null) {
                viewHolder.detectionValue.setText(String.format("%s%%", String.format(Locale.CANADA, "%.2f", (100 * model.getConfidence()))));
            }
        }

        return convertView;
    }
}
