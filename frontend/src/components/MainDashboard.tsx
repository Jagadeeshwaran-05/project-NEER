import React, { useState, useEffect } from "react";
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  CircularProgress,
  Tabs,
  Tab,
} from "@mui/material";
import { MapContainer, TileLayer, GeoJSON } from "react-leaflet";
import { FeatureCollection } from "geojson";
import WaterHealthCard from "./WaterHealthCard";
import IndexChart from "./IndexChart";
import AlertsPanel from "./AlertsPanel";
import HistoricalTrends from "./HistoricalTrends";
import PollutionMappingPanel from "./PollutionMappingPanel";
import ChatAssistant from "./ChatAssistant";
import { getAllLakes, Lake } from "../services/apiService";
import "leaflet/dist/leaflet.css";

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const MainDashboard: React.FC = () => {
  const [lakes, setLakes] = useState<Lake[]>([]);
  const [selectedYear, setSelectedYear] = useState<number>(2024);
  const [selectedLake, setSelectedLake] = useState<Lake | null>(null);
  const [loading, setLoading] = useState(true);
  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    fetchLakeData();
  }, [selectedYear]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const fetchLakeData = async () => {
    try {
      setLoading(true);
      console.log('Fetching lake data for year:', selectedYear);
      const data = await getAllLakes(selectedYear);
      console.log('Received lake data:', data);
      setLakes(data);
    } catch (error) {
      console.error("Error fetching lake data:", error);
      // Add an alert to show the user what's wrong
      alert(`Error fetching lake data: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const getHealthColor = (health: string) => {
    const colors = {
      Excellent: "#1e40af",
      Good: "#3b82f6",
      Moderate: "#eab308",
      Poor: "#f97316",
      "Very Poor": "#dc2626",
    };
    return colors[health as keyof typeof colors] || "#6b7280";
  };

  const centerCoordinates: [number, number] = [10.981, 76.955];

  if (loading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="80vh"
      >
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ padding: 3 }}>
      <Typography variant="h4" gutterBottom>
        Project NEER - Lake Water Health Monitoring Dashboard
      </Typography>

      {/* Debug Info */}
      <Typography variant="body2" color="textSecondary" gutterBottom>
        Loaded {lakes.length} lakes for year {selectedYear}
        {lakes.length === 0 && !loading && " - No data available. Check backend connection."}
      </Typography>

      <Grid container spacing={3}>
        {/* Year Selection */}
        <Grid item xs={12} md={3}>
          <FormControl fullWidth>
            <InputLabel>Select Year</InputLabel>
            <Select
              value={selectedYear}
              onChange={(e) => setSelectedYear(e.target.value as number)}
              label="Select Year"
            >
              {Array.from({ length: 10 }, (_, i) => 2024 - i).map((year) => (
                <MenuItem key={year} value={year}>
                  {year}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* Interactive Map */}
        <Grid item xs={12} md={9}>
          <Card style={{ height: "500px" }}>
            <MapContainer
              center={centerCoordinates}
              zoom={13}
              style={{ height: "100%", width: "100%" }}
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              />

              {lakes.map((lake) => (
                <GeoJSON
                  key={lake.id}
                  data={lake.geometry}
                  style={{
                    fillColor: getHealthColor(lake.waterHealth),
                    fillOpacity: 0.6,
                    color: getHealthColor(lake.waterHealth),
                    weight: 2,
                  }}
                  onEachFeature={(feature, layer) => {
                    layer.on("click", () => setSelectedLake(lake));
                  }}
                />
              ))}
            </MapContainer>
          </Card>
        </Grid>

        {/* Lake Cards Section - Show all lakes */}
        {lakes.length > 0 && (
          <Grid item xs={12}>
            <Typography variant="h5" gutterBottom>
              Lake Water Quality Overview
            </Typography>
            <Grid container spacing={2}>
              {lakes.map((lake) => (
                <Grid item xs={12} md={6} lg={4} key={lake.id}>
                  <WaterHealthCard lake={lake} />
                </Grid>
              ))}
            </Grid>
          </Grid>
        )}

        {/* Advanced Analytics Tabs */}
        <Grid item xs={12}>
          <Card>
            <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
              <Tabs value={tabValue} onChange={handleTabChange} aria-label="lake analytics tabs">
                <Tab label="Water Quality Charts" />
                <Tab label="Historical Trends" />
                <Tab label="Quality Alerts" />
                <Tab label="Pollution Mapping" />
              </Tabs>
            </Box>
            
            <TabPanel value={tabValue} index={0}>
              <Grid container spacing={3}>
                {/* Lake Information Panel - Selected Lake Details */}
                {selectedLake && (
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>
                      Selected Lake Details
                    </Typography>
                    <WaterHealthCard lake={selectedLake} />
                  </Grid>
                )}

                {/* Charts Section */}
                <Grid item xs={12} md={selectedLake ? 6 : 12}>
                  <IndexChart lakes={lakes} selectedYear={selectedYear} />
                </Grid>
              </Grid>
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              <HistoricalTrends lakes={lakes} />
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              <AlertsPanel />
            </TabPanel>

            <TabPanel value={tabValue} index={3}>
              <PollutionMappingPanel lakes={lakes} />
            </TabPanel>
          </Card>
        </Grid>
      </Grid>

      <ChatAssistant
        lakes={lakes}
        selectedLake={selectedLake}
        selectedYear={selectedYear}
      />
    </Box>
  );
};

export default MainDashboard;
