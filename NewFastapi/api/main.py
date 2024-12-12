from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from datetime import datetime
import json
import httpx
import logging
import re
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI client
def init_openai():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("Warning: OPENAI_API_KEY not found")
        return None
    return OpenAI(api_key=api_key)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
client = init_openai()

# Model definitions
class FinancialGoal(BaseModel):
    goal_type: str
    target_amount: float
    timeline_years: int

class UserProfile(BaseModel):
    # Personal Information
    age: int
    employment_status: str
    annual_income: float
    marital_status: str
    dependents: int
    
    # Financial Goals
    financial_goals: List[FinancialGoal]
    investment_horizon: str  # Short-term, Medium-term, Long-term
    
    # Risk Profile
    risk_tolerance: str  # Low, Moderate, High
    comfort_with_fluctuations: int  # 1-10
    
    # Current Financial Status
    monthly_income: float
    monthly_expenses: float
    existing_debts: Optional[str]
    emergency_fund_months: Optional[int]
    
    # Investment Preferences
    investment_preferences: List[str]
    management_style: str  # Active, Passive, Not Sure
    ethical_criteria: Optional[str]
    tax_advantaged_options: bool
    
    # Additional Preferences
    liquidity_needs: str  # Yes/No with timeframe
    investment_knowledge: str  # Beginner, Intermediate, Advanced
    previous_investments: Optional[str]
    involvement_level: str  # Hands-on, Hands-off, Collaborative
    major_life_changes: Optional[bool] = False
    life_change_details: Optional[str] = None

class InvestmentSuggestion(BaseModel):
    investment_type: str
    allocation_percentage: float
    details: str
    specific_suggestions: List[Dict[str, Any]]
    entry_strategy: str
    exit_strategy: str
    risk_mitigation: str

class PortfolioRecommendation(BaseModel):
    explanation: str
    recommendations: List[InvestmentSuggestion]
    market_analysis: Dict[str, Dict[str, Any]]
    review_schedule: str
    disclaimer: str

def analyze_user_life_stage(user_data: UserProfile) -> Dict[str, Any]:
    """Analyze user's life stage and financial situation."""
    life_stage = {
        "stage": "",
        "priorities": [],
        "constraints": [],
        "opportunities": []
    }
    
    # Determine life stage
    if user_data.age < 30:
        life_stage["stage"] = "Early Career"
        life_stage["priorities"] = ["Building emergency fund", "Career growth", "Long-term wealth creation"]
        life_stage["opportunities"] = ["Higher risk tolerance", "Time in market", "Human capital growth"]
    elif user_data.age < 45:
        life_stage["stage"] = "Mid Career"
        life_stage["priorities"] = ["Family needs", "Education planning", "Retirement saving"]
        life_stage["opportunities"] = ["Peak earning years", "Investment diversification"]
    else:
        life_stage["stage"] = "Pre-Retirement"
        life_stage["priorities"] = ["Retirement readiness", "Wealth preservation", "Legacy planning"]
        life_stage["opportunities"] = ["Catch-up contributions", "Tax optimization"]
    
    # Add family considerations
    if user_data.dependents > 0:
        life_stage["priorities"].append("Family protection")
        life_stage["constraints"].append("Higher monthly expenses")
    
    # Analyze financial health
    monthly_savings = user_data.monthly_income - user_data.monthly_expenses
    savings_rate = (monthly_savings / user_data.monthly_income) * 100
    
    if savings_rate < 20:
        life_stage["constraints"].append("Limited savings capacity")
    else:
        life_stage["opportunities"].append("Strong saving potential")
    
    if user_data.existing_debts:
        life_stage["constraints"].append("Debt management needed")
    
    return life_stage

def analyze_financial_health(user_data: UserProfile) -> Dict[str, Any]:
    """Comprehensive analysis of user's financial health."""
    monthly_income = user_data.monthly_income
    monthly_expenses = user_data.monthly_expenses
    savings_potential = monthly_income - monthly_expenses
    
    health_metrics = {
        "savings_ratio": (savings_potential / monthly_income) * 100,
        "emergency_fund_status": "Adequate" if user_data.emergency_fund_months >= 6 else "Inadequate",
        "debt_status": "Present" if user_data.existing_debts else "None",
        "investment_capacity": savings_potential * 0.6,  # 60% of savings for investments
        "risk_factors": [],
        "opportunities": []
    }
    
    # Analyze metrics
    if health_metrics["savings_ratio"] < 20:
        health_metrics["risk_factors"].append("Low savings rate")
    else:
        health_metrics["opportunities"].append("Strong saving potential")
    
    if user_data.existing_debts:
        health_metrics["risk_factors"].append("Existing debt obligations")
    
    if health_metrics["emergency_fund_status"] == "Inadequate":
        health_metrics["risk_factors"].append("Insufficient emergency fund")
    
    return health_metrics

def analyze_investment_preferences(preferences: List[str]) -> Dict[str, Any]:
    """Analyze user's investment preferences."""
    preference_profile = {
        "has_preferences": bool(preferences),
        "preferred_types": set(preferences) if preferences else set(),
        "risk_level": "Unknown",
        "focus_areas": [],
        "diversification_needed": True
    }
    
    if preferences:
        # Determine risk level from preferences
        high_risk = {"Stocks", "Cryptocurrencies", "Real Estate"}
        moderate_risk = {"Mutual Funds", "ETFs", "ELSS"}
        low_risk = {"Bonds", "Fixed Deposits", "Gold", "PPF", "EPF", "NPS"}
        
        preference_set = set(preferences)
        if preference_set.intersection(high_risk):
            preference_profile["risk_level"] = "High"
        elif preference_set.intersection(moderate_risk):
            preference_profile["risk_level"] = "Moderate"
        elif preference_set.intersection(low_risk):
            preference_profile["risk_level"] = "Low"
        
        # Map preferences to focus areas
        focus_mappings = {
            "Stocks": "Equity",
            "Mutual Funds": "Diversified",
            "ETFs": "Index-based",
            "Bonds": "Fixed Income",
            "Fixed Deposits": "Fixed Income",
            "Gold": "Commodities",
            "Real Estate": "Real Assets",
            "ELSS": "Tax Saving",
            "PPF": "Tax Saving",
            "EPF": "Retirement",
            "NPS": "Retirement",
            "Cryptocurrencies": "Alternative"
        }
        
        preference_profile["focus_areas"] = [
            focus_mappings[pref] for pref in preferences 
            if pref in focus_mappings
        ]
        
        preference_profile["diversification_needed"] = len(preference_profile["focus_areas"]) < 3
    
    return preference_profile

def analyze_tax_efficiency(user_data: UserProfile) -> Dict[str, Any]:
    """Analyze tax efficiency opportunities."""
    tax_profile = {
        "tax_saving_interest": user_data.tax_advantaged_options,
        "recommended_tax_instruments": [],
        "annual_tax_saving_potential": 0.0
    }
    
    if user_data.tax_advantaged_options:
        tax_profile["recommended_tax_instruments"] = [
            {"instrument": "ELSS", "rationale": "Tax saving with equity exposure"},
            {"instrument": "PPF", "rationale": "Long-term tax-free returns"},
            {"instrument": "NPS", "rationale": "Additional tax benefits for retirement"}
        ]
        
        # Calculate potential tax savings (80C limit: 1.5 lakhs)
        annual_income = user_data.annual_income
        if annual_income > 1000000:  # 10 lakhs
            tax_profile["annual_tax_saving_potential"] = 150000
        else:
            tax_profile["annual_tax_saving_potential"] = min(annual_income * 0.15, 150000)
    
    return tax_profile

def get_asset_allocation(
    risk_tolerance: str,
    investment_horizon: str,
    age: int
) -> Dict[str, float]:
    """Determine base asset allocation based on risk profile and age."""
    # Base allocations by risk tolerance
    allocations = {
        "Conservative": {
            "Stocks": 20.0,
            "Mutual_Funds": 25.0,
            "Bonds": 40.0,
            "Gold": 15.0
        },
        "Moderate": {
            "Stocks": 35.0,
            "Mutual_Funds": 30.0,
            "Bonds": 25.0,
            "Gold": 10.0
        },
        "Aggressive": {
            "Stocks": 50.0,
            "Mutual_Funds": 30.0,
            "Bonds": 15.0,
            "Gold": 5.0
        }
    }
    
    # Get base allocation
    base = allocations.get(risk_tolerance, allocations["Moderate"]).copy()
    
    # Adjust for age (reduce equity exposure as age increases)
    if age > 45:
        equity_reduction = min((age - 45) * 0.5, 10.0)  # Reduce up to 10%
        base["Stocks"] = max(base["Stocks"] - equity_reduction, 10.0)
        base["Bonds"] += equity_reduction
    
    # Adjust for investment horizon
    if investment_horizon == "Short-term":
        # Reduce volatile assets for short-term
        base["Stocks"] *= 0.7
        base["Bonds"] += (base["Stocks"] * 0.3)
    elif investment_horizon == "Long-term":
        # Increase growth assets for long-term
        if age < 40:  # Only for younger investors
            base["Stocks"] *= 1.2
            base["Bonds"] -= (base["Stocks"] * 0.2)
    
    # Normalize to ensure 100%
    total = sum(base.values())
    return {k: round(v * 100 / total, 1) for k, v in base.items()}

def get_investment_recommendations(market_data: Dict, user_data: UserProfile) -> List[Dict]:
    """Generate personalized investment recommendations."""
    try:
        # Initial analysis
        life_stage = analyze_user_life_stage(user_data)
        financial_health = analyze_financial_health(user_data)
        tax_profile = analyze_tax_efficiency(user_data)
        preference_profile = analyze_investment_preferences(user_data.investment_preferences)
        
        recommendations = []
        
        # Emergency Fund Check
        if financial_health["emergency_fund_status"] == "Inadequate":
            emergency_allocation = min(30.0, max(10.0, 100 - financial_health["savings_ratio"]))
            recommendations.append({
                "investment_type": "Emergency_Fund",
                "allocation_percentage": emergency_allocation,
                "details": "Priority allocation for emergency fund",
                "specific_suggestions": [
                    {
                        "name": "High-yield Savings",
                        "ticker": "N/A",
                        "rationale": "Immediate liquidity"
                    },
                    {
                        "name": "Liquid Funds",
                        "ticker": "Various",
                        "rationale": "Better returns with high liquidity"
                    }
                ],
                "entry_strategy": "Regular monthly transfers",
                "exit_strategy": "Maintain 6 months coverage",
                "risk_mitigation": "Spread across multiple banks"
            })
        
        # Debt Management Check
        if user_data.existing_debts:
            debt_allocation = min(25.0, max(10.0, financial_health["savings_ratio"] * 0.4))
            recommendations.append({
                "investment_type": "Debt_Management",
                "allocation_percentage": debt_allocation,
                "details": "Debt reduction strategy",
                "specific_suggestions": [
                    {
                        "name": "High-interest Debt",
                        "ticker": "N/A",
                        "rationale": "Priority repayment"
                    }
                ],
                "entry_strategy": "Accelerated debt repayment",
                "exit_strategy": "Continue till debt-free",
                "risk_mitigation": "Focus on highest interest debt first"
            })
        
        # Get base asset allocation
        base_allocation = get_asset_allocation(
            user_data.risk_tolerance,
            user_data.investment_horizon,
            user_data.age
        )
        
        # Calculate remaining allocation
        allocated = sum(rec["allocation_percentage"] for rec in recommendations)
        remaining = 100 - allocated
        
        # Adjust base allocations to remaining percentage
        adjusted_allocation = {
            k: v * remaining / 100 
            for k, v in base_allocation.items()
        }
        
        # Generate recommendations for each asset type
        for asset_type, allocation in adjusted_allocation.items():
            market_info = market_data.get(asset_type, {})
            
            recommendations.append({
                "investment_type": asset_type,
                "allocation_percentage": round(allocation, 1),
                "details": get_personalized_details(
                    asset_type,
                    user_data,
                    life_stage,
                    preference_profile
                ),
                "specific_suggestions": customize_suggestions(
                    asset_type,
                    user_data,
                    market_info.get("specific_suggestions", []),
                    preference_profile
                ),
                "entry_strategy": get_entry_strategy(
                    asset_type,
                    user_data,
                    preference_profile
                ),
                "exit_strategy": get_exit_strategy(
                    asset_type,
                    user_data,
                    preference_profile
                ),
                "risk_mitigation": get_risk_mitigation(
                    asset_type,
                    user_data,
                    financial_health
                )
            })
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error in recommendations: {str(e)}")
        return []

def get_personalized_details(
    asset_type: str,
    user_data: UserProfile,
    life_stage: Dict,
    preference_profile: Dict
) -> str:
    """Generate personalized investment details."""
    details = f"Recommended allocation for {asset_type.replace('_', ' ').lower()} "
    
    # Add life stage context
    details += f"aligned with your {life_stage['stage']} life stage"
    
    # Add preference context
    if preference_profile["has_preferences"]:
        if asset_type.replace("_", " ") in preference_profile["preferred_types"]:
            details += ", matching your investment preferences"
    
    # Add goal-based context
    if user_data.investment_horizon == "Short-term":
        details += ", focused on capital preservation"
    elif user_data.investment_horizon == "Long-term":
        details += ", targeting long-term growth"
    
    # Add knowledge-based context
    if user_data.investment_knowledge == "Beginner":
        details += ", with straightforward investment options"
    elif user_data.investment_knowledge == "Advanced":
        details += ", including sophisticated strategies"
    
    return details

def customize_suggestions(
    asset_type: str,
    user_data: UserProfile,
    market_suggestions: List[Dict],
    preference_profile: Dict
) -> List[Dict]:
    """Customize investment suggestions based on user profile."""
    suggestions = []
    
    # Add preference-based suggestions
    if preference_profile["has_preferences"]:
        if asset_type == "Stocks":
            if user_data.risk_tolerance == "Aggressive":
                suggestions.append({
                    "name": "Growth Stocks Portfolio",
                    "ticker": "Various",
                    "rationale": "High growth potential for aggressive investors"
                })
            else:
                suggestions.append({
                    "name": "Blue-chip Stocks",
                    "ticker": "Various",
                    "rationale": "Stable, established companies"
                })
        
        elif asset_type == "Mutual_Funds":
            if user_data.tax_advantaged_options:
                suggestions.append({
                    "name": "ELSS Funds",
                    "ticker": "Various",
                    "rationale": "Tax benefits with equity exposure"
                })
            if user_data.risk_tolerance == "Conservative":
                suggestions.append({
                    "name": "Balanced Advantage Funds",
                    "ticker": "Various",
                    "rationale": "Dynamic asset allocation for stability"
                })
    
    # Add market suggestions if needed
    for suggestion in market_suggestions:
        if len(suggestions) < 3:  # Keep max 3 suggestions
            suggestions.append(suggestion)
    
    return suggestions[:3]

def get_entry_strategy(
    asset_type: str,
    user_data: UserProfile,
    preference_profile: Dict
) -> str:
    """Generate personalized entry strategy."""
    strategy = ""
    
    base_strategies = {
        "Stocks": {
            "Conservative": "Staggered buying through SIP",
            "Moderate": "Mix of lump sum and SIP",
            "Aggressive": "Strategic buying on market dips"
        },
        "Mutual_Funds": {
            "Conservative": "Fixed monthly SIP",
            "Moderate": "SIP with occasional top-ups",
            "Aggressive": "Dynamic SIP based on market conditions"
        },
        "Bonds": {
            "Conservative": "Ladder strategy with government bonds",
            "Moderate": "Mix of government and corporate bonds",
            "Aggressive": "Focus on higher-yield corporate bonds"
        },
        "Gold": {
            "Conservative": "Small, regular purchases",
            "Moderate": "Quarterly purchases",
            "Aggressive": "Strategic buying on price dips"
        }
    }
    
    strategy = base_strategies.get(asset_type, {}).get(
        user_data.risk_tolerance,
        "Systematic investment approach"
    )
    
    # Add user knowledge context
    if user_data.investment_knowledge == "Beginner":
        strategy += " with guided assistance"
    elif user_data.investment_knowledge == "Advanced":
        strategy += " with flexibility for tactical adjustments"
    
    return strategy

def get_exit_strategy(
    asset_type: str,
    user_data: UserProfile,
    preference_profile: Dict
) -> str:
    """Generate personalized exit strategy."""
    strategy = ""
    
    base_strategies = {
        "Stocks": {
            "Short-term": "Regular review and profit booking",
            "Medium-term": "Hold with trailing stop losses",
            "Long-term": "Buy and hold with annual rebalancing"
        },
        "Mutual_Funds": {
            "Short-term": "Monitor and switch based on performance",
            "Medium-term": "Review quarterly and rebalance",
            "Long-term": "Stay invested with annual review"
        },
        "Bonds": {
            "Short-term": "Hold till maturity",
            "Medium-term": "Reinvest on better yields",
            "Long-term": "Ladder maturity dates"
        },
        "Gold": {
            "Short-term": "Book profits on significant rises",
            "Medium-term": "Hold as portfolio hedge",
            "Long-term": "Maintain as strategic allocation"
        }
    }
    
    strategy = base_strategies.get(asset_type, {}).get(
        user_data.investment_horizon,
        "Regular review and rebalancing"
    )
    
    # Add liquidity needs context
    if user_data.liquidity_needs == "Yes":
        strategy += ", maintaining necessary liquidity"
    
    return strategy

def get_risk_mitigation(
    asset_type: str,
    user_data: UserProfile,
    financial_health: Dict
) -> str:
    """Generate comprehensive risk mitigation strategy."""
    strategies = []
    
    # Base risk mitigation by asset type
    base_strategies = {
        "Stocks": "Diversification across sectors and market caps",
        "Mutual_Funds": "Mix of different fund categories",
        "Bonds": "Ladder maturities and mix credit ratings",
        "Gold": "Mix of physical and paper gold"
    }
    strategies.append(base_strategies.get(asset_type, "Diversification"))
    
    # Add financial health considerations
    if financial_health["risk_factors"]:
        strategies.append("regular monitoring")
    
    # Add dependents consideration
    if user_data.dependents > 0:
        strategies.append("focus on capital preservation")
    
    # Add experience level consideration
    if user_data.investment_knowledge == "Beginner":
        strategies.append("start with lower-risk options")
    
    return ", ".join(strategies)
async def get_market_analysis(asset_type: str) -> Dict:
    """Get real-time market analysis using Perplexity API."""
    try:
        if not PERPLEXITY_API_KEY:
            logger.warning("No Perplexity API key found, using fallback data")
            return get_fallback_market_data(asset_type)

        url = "https://api.perplexity.ai/chat/completions"
        
        # Different queries for different asset types
        queries = {
            "Stocks": """Analyze current Indian stock market including:
                1. Today's top performing stocks in BSE/NSE
                2. Current market sentiment and trends
                3. Latest market risks and opportunities
                4. Specific stock recommendations
                Response must be JSON with current_trend, outlook, key_factors, risks, and specific_suggestions.""",
            
            "Mutual_Funds": """Analyze current Indian mutual fund market including:
                1. Top performing mutual funds this month
                2. Current AUM trends
                3. Latest fund performance data
                4. Specific fund recommendations
                Response must be JSON with current_trend, outlook, key_factors, risks, and specific_suggestions.""",
            
            "Bonds": """Analyze current Indian bond market including:
                1. Latest bond yields
                2. Government and corporate bond performance
                3. Current market opportunities
                4. Specific bond recommendations
                Response must be JSON with current_trend, outlook, key_factors, risks, and specific_suggestions.""",
            
            "Gold": """Analyze current gold market including:
                1. Latest gold prices and trends
                2. Market outlook and forecasts
                3. Investment opportunities
                4. Specific recommendations for gold investments
                Response must be JSON with current_trend, outlook, key_factors, risks, and specific_suggestions."""
        }

        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a financial analyst. Provide real-time market analysis in this exact JSON format:
                    {
                        "current_trend": "detailed current market description",
                        "outlook": "detailed market outlook",
                        "key_factors": ["list of key market factors"],
                        "risks": ["list of key risks"],
                        "specific_suggestions": [
                            {
                                "name": "investment name",
                                "ticker": "ticker symbol",
                                "rationale": "detailed rationale with current data"
                            }
                        ]
                    }"""
                },
                {"role": "user", "content": queries.get(asset_type, "")}
            ],
            "max_tokens": 2048
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' in result and result['choices']:
                content = result['choices'][0]['message']['content']
                try:
                    # Clean up the content
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    
                    # Parse JSON response
                    market_data = json.loads(content)
                    
                    # Validate required fields
                    required_fields = ['current_trend', 'outlook', 'key_factors', 'risks', 'specific_suggestions']
                    if all(field in market_data for field in required_fields):
                        return market_data
                    
                except json.JSONDecodeError:
                    logger.error("Error parsing market analysis response")
                except Exception as e:
                    logger.error(f"Error processing market data: {str(e)}")
            
            return get_fallback_market_data(asset_type)

    except Exception as e:
        logger.error(f"Error fetching market analysis: {str(e)}")
        return get_fallback_market_data(asset_type)

def get_fallback_market_data(asset_type: str) -> Dict:
    """Provide fallback market data when API fails."""
    fallback_data = {
        "Stocks": {
            "current_trend": "Mixed trends in equity markets",
            "outlook": "Cautiously optimistic outlook",
            "key_factors": [
                "Global market conditions",
                "Domestic economic growth",
                "Corporate earnings"
            ],
            "risks": [
                "Market volatility",
                "Economic uncertainty",
                "Global factors"
            ],
            "specific_suggestions": [
                {
                    "name": "Large Cap Stock Fund",
                    "ticker": "NIFTY50",
                    "rationale": "Based on index performance"
                },
                {
                    "name": "Blue-chip Companies",
                    "ticker": "Various",
                    "rationale": "Stable, established companies"
                }
            ]
        },
        # ... [Keep your existing fallback data for other asset types]
    }
    
    return fallback_data.get(asset_type, fallback_data.get("Stocks", {}))

# Update the analyze_portfolio endpoint
@app.post("/analyze-portfolio", response_model=PortfolioRecommendation)
async def analyze_portfolio(user_data: UserProfile) -> PortfolioRecommendation:
    """Generate personalized portfolio recommendation."""
    try:
        logger.info(f"Starting portfolio analysis for user age {user_data.age}")
        
        # Get market analysis for all asset types
        market_analysis = {}
        asset_types = ["Stocks", "Mutual_Funds", "Bonds", "Gold"]
        
        # Get market analysis for each asset type
        for asset in asset_types:
            market_analysis[asset] = await get_market_analysis(asset)
        
        # Generate recommendations
        recommendations = get_investment_recommendations(market_data=market_analysis, user_data=user_data)
        
        # Rest of your analyze_portfolio function...
        # Generate explanation
        life_stage = analyze_user_life_stage(user_data)
        financial_health = analyze_financial_health(user_data)
        
        explanation = []
        explanation.append(
            f"Based on your profile as a {user_data.age}-year-old {user_data.investment_knowledge.lower()}-level investor "
            f"in the {life_stage['stage']} stage, we have prepared a personalized investment strategy. "
            f"Your {user_data.risk_tolerance.lower()} risk tolerance and {user_data.investment_horizon.lower()} "
            f"investment horizon have been key factors in these recommendations."
        )
        
        # Add financial health context
        if financial_health["risk_factors"]:
            explanation.append("\nKey Financial Considerations:")
            for factor in financial_health["risk_factors"]:
                explanation.append(f"• {factor}")
        
        # Add allocation explanation
        explanation.append("\nRecommended Allocations:")
        for rec in recommendations:
            explanation.append(
                f"• {rec['investment_type']}: {rec['allocation_percentage']}% - {rec['details']}"
            )
        
        # Determine review schedule
        review_schedule = (
            "Monthly" if user_data.investment_knowledge == "Beginner"
            else "Quarterly" if user_data.investment_horizon == "Short-term"
            else "Semi-annually"
        )
        
        portfolio = PortfolioRecommendation(
            explanation="\n".join(explanation),
            recommendations=recommendations,
            market_analysis=market_analysis,
            review_schedule=review_schedule,
            disclaimer=(
                "This recommendation is based on your profile and current market conditions. "
                "Past performance does not guarantee future results. "
                "Please consult with a financial advisor before making investment decisions."
            )
        )
        
        return portfolio
        
    except Exception as e:
        logger.error(f"Error in analyze_portfolio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating portfolio recommendation: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
