// SUSC-18E Curitiba - Sentinel-1 SAR tecnico somente revisao
// AOI derivada dos 43 patches CUR. Nao e geometria oficial de ocorrencia.

var eventId = 'S17C_REF_0060';
var eventPublicId = 'CUR_2022_01_15';
var aoi = ee.Geometry.Polygon([[[-49.403546858188726, -25.599740077698588], [-49.08763058997096, -25.599740077698588], [-49.08763058997096, -25.29008810702265], [-49.403546858188726, -25.29008810702265], [-49.403546858188726, -25.599740077698588]]]);

var collection = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('resolution_meters', 10));

var pre = collection
  .filterDate('2021-12-16', '2022-01-14')
  .select(['VV', 'VH'])
  .mean()
  .clip(aoi);

var post = collection
  .filterDate('2022-01-15', '2022-02-14')
  .select(['VV', 'VH'])
  .mean()
  .clip(aoi);

var deltaVv = post.select('VV').subtract(pre.select('VV')).rename('delta_vv_db');
var deltaVh = post.select('VH').subtract(pre.select('VH')).rename('delta_vh_db');
var waterCandidate = deltaVv.lt(-1.5).and(deltaVh.lt(-1.5)).rename('sar_water_candidate');
var stack = pre.addBands(post).addBands(deltaVv).addBands(deltaVh).addBands(waterCandidate);

Map.centerObject(aoi, 11);
Map.addLayer(aoi, {color: 'yellow'}, 'AOI tecnica 43 patches CUR');
Map.addLayer(deltaVv, {min: -5, max: 5, palette: ['blue', 'white', 'red']}, 'Delta VV dB');
Map.addLayer(waterCandidate.selfMask(), {palette: ['00FFFF']}, 'Candidato agua SAR');

Export.image.toDrive({
  image: stack,
  description: 'S18E_CUR_2022_01_15_Sentinel1_stack',
  folder: 'REV_P_SUSC_18E',
  fileNamePrefix: 's18e_curitiba_2022_01_15_sentinel1_stack',
  region: aoi,
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.table.toDrive({
  collection: ee.FeatureCollection([ee.Feature(aoi, {
    candidate_event_id: eventId,
    evento_publico: eventPublicId,
    uso_permitido: 'footprint_tecnico_somente_revisao',
    geometria_oficial_de_ocorrencia: false,
    ground_truth: false,
    eligible_for_training: false,
    score_v7_allowed: false
  })]),
  description: 'S18E_CUR_2022_01_15_AOI_manifest',
  folder: 'REV_P_SUSC_18E',
  fileNamePrefix: 's18e_curitiba_2022_01_15_aoi_manifest',
  fileFormat: 'CSV'
});
