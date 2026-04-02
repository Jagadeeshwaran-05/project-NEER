// Use environment variable for API URL, fallback to localhost for development
const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

export interface Lake {
  id: string;
  name: string;
  ndwi: number;
  ndci: number;
  fai: number;
  mci: number;
  swir_ratio: number;
  turbidity: number;
  bodLevel: number;
  waterHealth: string;
  pollutionCauses: string;
  suggestions: string;
  geometry: any;
  year: number;
}

export interface HistoricalData {
  year: number;
  ndwi: number;
  ndci: number;
  fai: number;
  mci: number;
  bodLevel: number;
  waterHealth: string;
  trend: string;
  turbidity: number;
  swir_ratio: number;
}

export interface HistoricalResponse {
  historical_data: HistoricalData[];
  trend_analysis: {
    overall_trend: string;
    trend_counts: {
      improving: number;
      degrading: number;
      stable: number;
    };
    data_points: number;
  };
}

export interface Alert {
  id: string;
  lake_name: string;
  alert_type: string;
  severity: string;
  message: string;
  timestamp: string;
  current_bod?: number;
  previous_bod?: number;
  change?: number;
  recommended_action: string;
}

export interface AlertsResponse {
  alerts: Alert[];
  total_alerts: number;
  last_updated: string;
}

export interface PollutionSource {
  type: string;
  severity: string;
  distance_km: number;
}

export interface PollutionMapping {
  lake_name: string;
  catchment_analysis: {
    total_area_km2: number;
    urban_coverage_percent: number;
    industrial_coverage_percent: number;
    vegetation_coverage_percent: number;
    water_coverage_percent: number;
  };
  pollution_risk_score: number;
  risk_level: string;
  identified_sources: PollutionSource[];
  recommendations: string[];
}

export interface ChatContextLake {
  id: string;
  name: string;
  year: number;
  waterHealth: string;
  ndwi: number;
  ndci: number;
  fai: number;
  mci: number;
  swir_ratio: number;
  turbidity: number;
  bodLevel: number;
  pollutionCauses: string;
  suggestions: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  message: string;
  history: ChatMessage[];
  year?: number;
  lake?: ChatContextLake | null;
}

export interface ChatResponse {
  reply: string;
  source: string;
  model?: string | null;
  retrieved_count?: number;
  rag_ready?: boolean;
}

export interface AISuggestionRequest {
  id: string;
  name: string;
  year: number;
  waterHealth: string;
  ndwi: number;
  ndci: number;
  fai: number;
  mci: number;
  swir_ratio: number;
  turbidity: number;
  bodLevel: number;
  pollutionCauses: string;
  suggestions: string;
}

export interface AISuggestionResponse {
  suggestion: string;
  source: string;
  model?: string | null;
}

export const getAllLakes = async (year: number = 2024): Promise<Lake[]> => {
  const response = await fetch(`${API_BASE_URL}/lakes?year=${year}`);
  if (!response.ok) {
    throw new Error("Failed to fetch lakes data");
  }
  return response.json();
};

export const getLakeHistory = async (
  lakeId: string,
  startYear: number = 2020,
  endYear: number = 2024
): Promise<HistoricalResponse> => {
  const response = await fetch(
    `${API_BASE_URL}/lakes/${lakeId}/history?start_year=${startYear}&end_year=${endYear}`
  );
  if (!response.ok) {
    throw new Error("Failed to fetch lake history");
  }
  return response.json();
};

export const getWaterQualityAlerts = async (): Promise<AlertsResponse> => {
  const response = await fetch(`${API_BASE_URL}/alerts`);
  if (!response.ok) {
    throw new Error("Failed to fetch water quality alerts");
  }
  return response.json();
};

export const getPollutionSources = async (lakeId: string): Promise<PollutionMapping> => {
  const response = await fetch(`${API_BASE_URL}/pollution-sources/${lakeId}`);
  if (!response.ok) {
    throw new Error("Failed to fetch pollution sources");
  }
  return response.json();
};

export const sendChatMessage = async (payload: ChatRequest): Promise<ChatResponse> => {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error("Failed to send chat message");
  }
  return response.json();
};

export const getAISuggestion = async (payload: AISuggestionRequest): Promise<AISuggestionResponse> => {
  const response = await fetch(`${API_BASE_URL}/ai-suggestions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error("Failed to fetch AI suggestion");
  }
  return response.json();
};
