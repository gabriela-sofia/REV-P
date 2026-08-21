// SUSC-19B - Extracao MapBiomas por patch (Earth Engine).
// Somente revisao. Nao contem credenciais. Nao baixa raster pesado.
// Ajuste os parametros editaveis abaixo e exporte o CSV leve.

// ===== PARAMETROS EDITAVEIS =====
var COLLECTION_ASSET = 'projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1';
var YEAR = 2022;                 // ano de referencia MapBiomas
var SCALE = 30;                              // resolucao MapBiomas (m)
var PATCHES_ASSET = 'users/SEU_USUARIO/susc_patches_300';  // FeatureCollection com os 300 patches (patch_id, geometria)
var EXPORT_NAME = 'mapbiomas_patch_landcover_19b_export';

// Classes MapBiomas agregadas (editavel conforme legenda oficial):
var CLASSES_AGUA = [33, 31];                 // rio/lago/oceano e aquicultura
var CLASSES_SOLO_EXPOSTO = [23, 25];         // praia/duna e area nao vegetada
var CLASSES_URBANO = [24];                   // area urbanizada
var CLASSES_VEGETACAO = [1, 3, 4, 5, 6, 49, 10, 11, 12]; // formacoes naturais

// ===== CARGA =====
var patches = ee.FeatureCollection(PATCHES_ASSET);
var mapbiomas = ee.Image(COLLECTION_ASSET).select('classification_' + YEAR);

// ===== PROPORCAO DE CLASSES POR PATCH =====
function proporcoes(feature) {
  var hist = mapbiomas.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: feature.geometry(),
    scale: SCALE,
    maxPixels: 1e9
  }).get('classification_' + YEAR);
  hist = ee.Dictionary(hist);
  var total = ee.Number(hist.values().reduce(ee.Reducer.sum()));
  function prop(classes) {
    var soma = ee.List(classes).iterate(function(c, acc) {
      c = ee.Number(c).format();
      return ee.Number(acc).add(ee.Number(hist.get(c, 0)));
    }, 0);
    return ee.Number(soma).divide(total);
  }
  // classe majoritaria
  var keys = hist.keys();
  var vals = ee.Array(hist.values());
  var idxMax = vals.argmax().get(0);
  var classeMajoritaria = ee.Number.parse(keys.get(idxMax));
  return feature.set({
    'mapbiomas_year': YEAR,
    'mapbiomas_class_majority': classeMajoritaria,
    'water_prop': prop(CLASSES_AGUA),
    'exposed_soil_prop': prop(CLASSES_SOLO_EXPOSTO),
    'impervious_proxy': prop(CLASSES_URBANO),
    'vegetation_prop_mapbiomas': prop(CLASSES_VEGETACAO),
    'class_distribution_json': hist,
    'pixel_count': total,
    'review_only': true
  });
}

var resultado = patches.map(proporcoes);

// ===== EXPORT CSV LEVE (sem raster) =====
Export.table.toDrive({
  collection: resultado,
  description: EXPORT_NAME,
  fileFormat: 'CSV',
  selectors: ['patch_id', 'mapbiomas_year', 'mapbiomas_class_majority', 'water_prop',
              'exposed_soil_prop', 'impervious_proxy', 'vegetation_prop_mapbiomas',
              'class_distribution_json', 'pixel_count', 'review_only']
});
