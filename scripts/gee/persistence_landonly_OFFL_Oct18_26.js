// Date modified October 18, 2026
// Rebuild workflow for Figure 3: NO2 persistence frequency panels (LAND-ONLY version)
//
// Changes from the June 11, 2026 version:
// - LAND_MASK_MODE: all monthly NO2 images are masked to land before any statistic,
//   so city-mean series, city-window thresholds and persistence frequencies use land
//   pixels only. Applied identically to all four cities.
// - Water mask source: JRC Global Surface Water 'occurrence' (1984-2021).
//   Cells with occurrence >= WATER_OCCURRENCE_THRESHOLD (%) are treated as water.
//   Cells with no GSW observation are treated as land.
// - Exports the monthly land-only city-mean series per city (CSV) so the M3 and
//   persistence tables can be regenerated from the same masked data.
//
// Threshold modes:
// - 'local_pixel_p75': each pixel uses its own 75th percentile through time.
// - 'city_window_p75': one city-wide scalar threshold from the monthly land-only
//   city-mean series over the 30 km buffer, then applied spatially.
//
// Display scaling modes:
// - 'fixed': use DISPLAY_MIN and DISPLAY_MAX directly.
// - 'observed_quantiles': common scale from pooled city percentiles.

var START_DATE = ee.Date('2019-01-01');
var END_DATE = ee.Date('2024-12-31');
var CITY_BUFFER_KM = 30;
var PANEL_RADIUS_KM = 60;
var EXPORT_SCALE_M = 1000;
var EXPORT_FOLDER = 'Pers-Land-Only';
var ENABLE_EXPORTS = true;

var LAND_MASK_MODE = 'jrc_water_occurrence'; // 'jrc_water_occurrence' or 'none'
var WATER_OCCURRENCE_THRESHOLD = 50;          // percent

var THRESHOLD_MODE = 'city_window_p75'; // 'local_pixel_p75' or 'city_window_p75'
var DISPLAY_SCALING_MODE = 'observed_quantiles'; // 'observed_quantiles' or 'fixed'
var DISPLAY_MIN = 0.0;
var DISPLAY_MAX = 0.6;
var DISPLAY_Q_LOW = 5;
var DISPLAY_Q_HIGH = 95;

var PERSISTENCE_PALETTE = [
  '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
  '#6baed6', '#4292c6', '#2171b5', '#084594'
];

var cityPoints = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([3.3792, 6.5244]), {
    city: 'Lagos', city_key: 'Lagos', country: 'Nigeria',
    region: 'West Africa', setting: 'Coastal'
  }),
  ee.Feature(ee.Geometry.Point([15.2663, -4.4419]), {
    city: 'Kinshasa', city_key: 'Kinshasa', country: 'Democratic Republic of the Congo',
    region: 'Central Africa', setting: 'Inland'
  }),
  ee.Feature(ee.Geometry.Point([39.2083, -6.7924]), {
    city: 'Dar es Salaam', city_key: 'Dar_es_Salaam', country: 'Tanzania',
    region: 'East Africa', setting: 'Coastal'
  }),
  ee.Feature(ee.Geometry.Point([28.0473, -26.2041]), {
    city: 'Johannesburg', city_key: 'Johannesburg', country: 'South Africa',
    region: 'Southern Africa', setting: 'Inland'
  })
]);

var no2Collection = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2')
  .filterDate(START_DATE, END_DATE)
  .select('tropospheric_NO2_column_number_density');

// Land mask: 1 = land, 0 = water. Water = GSW occurrence >= threshold.
var gswOccurrence = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence');
var landMask = (LAND_MASK_MODE === 'jrc_water_occurrence')
  ? gswOccurrence.unmask(0).lt(WATER_OCCURRENCE_THRESHOLD)
  : ee.Image.constant(1);

var monthCount = END_DATE.difference(START_DATE, 'month');
var monthOffsets = ee.List.sequence(0, monthCount.subtract(1));

function buildMonthlyNo2Collection(region) {
  return ee.ImageCollection.fromImages(
    monthOffsets.map(function(offset) {
      offset = ee.Number(offset);
      var monthStart = START_DATE.advance(offset, 'month');
      var monthEnd = monthStart.advance(1, 'month');
      var image = no2Collection
        .filterDate(monthStart, monthEnd)
        .mean()
        .rename('no2')
        .clip(region)
        .updateMask(landMask.clip(region))   // land-only from here on
        .set({
          month_start: monthStart.format('YYYY-MM-dd'),
          month_label: monthStart.format('YYYY-MM')
        });
      return image;
    })
  );
}

function buildCitySeries(monthlyNo2, cityBuffer) {
  return ee.FeatureCollection(
    monthlyNo2.map(function(image) {
      var meanValue = image.reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: cityBuffer,
        scale: EXPORT_SCALE_M,
        bestEffort: true,
        maxPixels: 1e9
      }).get('no2');
      return ee.Feature(null, {
        month_label: image.get('month_label'),
        month_start: image.get('month_start'),
        city_mean_no2_landonly: meanValue
      });
    })
  );
}

function buildPersistenceImage(cityFeature) {
  var cityName = ee.String(cityFeature.get('city'));
  var cityKey = ee.String(cityFeature.get('city_key'));
  var cityPoint = cityFeature.geometry();
  var cityBuffer = cityPoint.buffer(CITY_BUFFER_KM * 1000);
  var panelRegion = cityPoint.buffer(PANEL_RADIUS_KM * 1000);
  var monthlyNo2 = buildMonthlyNo2Collection(panelRegion);
  var citySeries = buildCitySeries(monthlyNo2, cityBuffer);

  var persistenceImage;
  var thresholdDescription;
  var thresholdScalar = null;

  if (THRESHOLD_MODE === 'local_pixel_p75') {
    var thresholdImage = monthlyNo2.reduce(ee.Reducer.percentile([75])).rename('no2_p75');
    persistenceImage = monthlyNo2.map(function(image) {
      return image.gte(thresholdImage).rename('high_flag');
    }).mean().rename('persistence_frequency');
    thresholdDescription = 'Per-pixel 75th percentile through time (land only)';
  } else if (THRESHOLD_MODE === 'city_window_p75') {
    thresholdScalar = ee.Number(
      citySeries.reduceColumns(
        ee.Reducer.percentile([75]),
        ['city_mean_no2_landonly']
      ).get('p75')
    );
    persistenceImage = monthlyNo2.map(function(image) {
      return image.gte(thresholdScalar).rename('high_flag');
    }).mean().rename('persistence_frequency');
    thresholdDescription = 'City-wide 75th percentile of monthly land-only city-mean NO2';
  } else {
    throw new Error('Unsupported THRESHOLD_MODE: ' + THRESHOLD_MODE);
  }

  var stats = persistenceImage.reduceRegion({
    reducer: ee.Reducer.percentile([DISPLAY_Q_LOW, DISPLAY_Q_HIGH])
      .combine({reducer2: ee.Reducer.minMax(), sharedInputs: true}),
    geometry: panelRegion,
    scale: EXPORT_SCALE_M,
    bestEffort: true,
    maxPixels: 1e9
  });

  return ee.Dictionary({
    city: cityName,
    city_key: cityKey,
    persistence_image: persistenceImage,
    city_buffer: cityBuffer,
    panel_region: panelRegion,
    city_series: citySeries,
    threshold_mode: THRESHOLD_MODE,
    land_mask_mode: LAND_MASK_MODE,
    threshold_description: thresholdDescription,
    threshold_scalar: thresholdScalar,
    stats: stats
  });
}

var cityList = cityPoints.toList(cityPoints.size());
var results = cityList.map(function(cityFeature) {
  return buildPersistenceImage(ee.Feature(cityFeature));
});

var pooledStatsFeatures = ee.FeatureCollection(results.map(function(result) {
  result = ee.Dictionary(result);
  var stats = ee.Dictionary(result.get('stats'));
  return ee.Feature(null, {
    city: result.get('city'),
    q_low: stats.get('persistence_frequency_p' + DISPLAY_Q_LOW),
    q_high: stats.get('persistence_frequency_p' + DISPLAY_Q_HIGH),
    min_value: stats.get('persistence_frequency_min'),
    max_value: stats.get('persistence_frequency_max')
  });
}));

print('Land mask mode', LAND_MASK_MODE, 'water occurrence threshold %', WATER_OCCURRENCE_THRESHOLD);
print('Figure 3 rebuild threshold mode', THRESHOLD_MODE);
print('Per-city persistence stats for cautious display scaling', pooledStatsFeatures);

var pooledQlow = ee.Number(
  pooledStatsFeatures.reduceColumns(ee.Reducer.min(), ['q_low']).get('min')
);
var pooledQhigh = ee.Number(
  pooledStatsFeatures.reduceColumns(ee.Reducer.max(), ['q_high']).get('max')
);

var suggestedDisplayMin = DISPLAY_SCALING_MODE === 'observed_quantiles'
  ? pooledQlow.getInfo() : DISPLAY_MIN;
var suggestedDisplayMax = DISPLAY_SCALING_MODE === 'observed_quantiles'
  ? pooledQhigh.getInfo() : DISPLAY_MAX;

print('Suggested common display min', suggestedDisplayMin);
print('Suggested common display max', suggestedDisplayMax);

var displayParams = {
  min: suggestedDisplayMin,
  max: suggestedDisplayMax,
  palette: PERSISTENCE_PALETTE
};

Map.setOptions('SATELLITE');
Map.centerObject(cityPoints, 4);

for (var i = 0; i < 4; i++) {
  var result = ee.Dictionary(results.get(i));
  var cityName = ee.String(result.get('city')).getInfo();
  var cityKey = ee.String(result.get('city_key')).getInfo();
  Map.addLayer(
    ee.Image(result.get('persistence_image')),
    displayParams,
    cityName + ' persistence (' + THRESHOLD_MODE + ', ' + LAND_MASK_MODE + ')',
    false
  );
  Map.addLayer(
    ee.Geometry(result.get('city_buffer')),
    {color: 'yellow'},
    cityName + ' 30 km buffer',
    false
  );

  if (ENABLE_EXPORTS) {
    Export.image.toDrive({
      image: ee.Image(result.get('persistence_image')),
      description: cityKey + '_NO2_persistence_frequency_landonly',
      folder: EXPORT_FOLDER,
      fileNamePrefix: cityKey + '_NO2_persistence_frequency_landonly',
      region: ee.Geometry(result.get('panel_region')),
      scale: EXPORT_SCALE_M,
      maxPixels: 1e13
    });
    Export.table.toDrive({
      collection: ee.FeatureCollection(result.get('city_series')),
      description: cityKey + '_monthly_city_mean_landonly',
      folder: EXPORT_FOLDER,
      fileNamePrefix: cityKey + '_monthly_city_mean_landonly',
      fileFormat: 'CSV'
    });
  }
}

// Print the city-window thresholds used for the persistence maps
print('City-window p75 thresholds (mol m-2):',
  results.map(function(r) {
    r = ee.Dictionary(r);
    return ee.Dictionary({
      city: r.get('city'),
      threshold: r.get('threshold_scalar')
    });
  })
);