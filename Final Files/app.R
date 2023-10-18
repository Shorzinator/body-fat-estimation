# app.R
library(shiny)
library(ggplot2)
library(shinythemes)
library(caret)
library(shinyBS)

file = read.csv("https://raw.githubusercontent.com/Sreeja1391/BodyFatProject/main/preprocessed_data.csv",header = TRUE)

lm5.1 = lm(BODYFAT~AGE+ABDOMEN+WRIST,file)

lm5.2 = lm(BODYFAT~AGE+CHEST+ABDOMEN+WRIST+BMI,file)

lm5.3 = lm(BODYFAT~AGE+NECK+ABDOMEN+WRIST,file)

lm5.4 = lm(BODYFAT~AGE+CHEST+ABDOMEN+HIP+WRIST+BMI,file)

lm5.5 = lm(BODYFAT~ABDOMEN+WRIST,file)

lm5.6 = lm(BODYFAT~AGE+NECK+CHEST+ABDOMEN+HIP+WRIST+BMI,file)

get_model_info <- function(selected_model) {
  aic_bic <- paste("AIC:", AIC(selected_model), "\nBIC:", BIC(selected_model))
  rsquared <- paste("R-squared:", summary(selected_model)$r.squared)
  
  actual_data <- file$BODYFAT
  predicted_data <- predict(selected_model, newdata = file)
  #rmse_value <- RMSE(predicted_data, actual_data)
  #rmse <- paste("RMSE:", rmse_value)
  
  coefficients <- coef(selected_model)
  
  return(list(
    aic_bic = aic_bic,
    rsquared = rsquared,
    #rmse = rmse,
    coefficients = coefficients
  ))
}

ui <- fluidPage(
  titlePanel("Body Fat Model Comparison and Predictions"),
  theme = shinytheme("superhero"),
  tags$style(HTML("
    .well {
      border-radius: 10px; /* Adjust the value as needed */
    }
    .shiny-text-output {
      border-radius: 10px; /* Adjust the value as needed */
    }
  ")),
  sidebarLayout(
    sidebarPanel(
      wellPanel(
        selectInput("comparison_model", "Select Model for Comparison",
                    choices = c("Linear Model 2", "Linear Model 3", "Linear Model 4", "Linear Model 5", "Linear Model 6")
        )
      ),
      wellPanel(
        textInput("age", "Enter Age"),
        textInput("abdomen", "Enter Abdomen Circumference(cm)"),
        textInput("wrist", "Enter Wrist Circumference(cm)"),
        actionButton("predict_button", "Predict",  class = "btn-primary")
      )
    ),
    mainPanel(
      h4("Final Model Information"),
      verbatimTextOutput("fixed_model_aic_bic"),
      verbatimTextOutput("fixed_model_rsquared"),
      #verbatimTextOutput("fixed_model_rmse"),
      verbatimTextOutput("fixed_model_coefficients"),
      br(),
      h4("Comparison Model Information"),
      verbatimTextOutput("comparison_model_aic_bic"),
      verbatimTextOutput("comparison_model_rsquared"),
      #verbatimTextOutput("comparison_model_rmse"),
      verbatimTextOutput("comparison_model_coefficients"),
      br(),
      h4("Prediction Results"),
      verbatimTextOutput("prediction_result")
    )
  )
)


server <- function(input, output) {
  observe({
    fixed_info <- get_model_info(lm5.1)
    comparison_model <- switch(input$comparison_model,
                               "Linear Model 2" = lm5.2,
                               "Linear Model 3" = lm5.3,
                               "Linear Model 4" = lm5.4,
                               "Linear Model 5" = lm5.5,
                               "Linear Model 6" = lm5.6)
    
    comparison_info <- get_model_info(comparison_model)
    
    output$fixed_model_aic_bic <- renderText({ fixed_info$aic_bic })
    output$fixed_model_rsquared <- renderText({ fixed_info$rsquared })
    #output$fixed_model_rmse <- renderText({ fixed_info$rmse })
    output$fixed_model_coefficients <- renderPrint({ fixed_info$coefficients })
    
    output$comparison_model_aic_bic <- renderText({ comparison_info$aic_bic })
    output$comparison_model_rsquared <- renderText({ comparison_info$rsquared })
    #output$comparison_model_rmse <- renderText({ comparison_info$rmse })
    output$comparison_model_coefficients <- renderPrint({ comparison_info$coefficients })
  })
  
  observeEvent(input$predict_button, {
    if (any(sapply(c(input$age, input$abdomen, input$wrist), function(x) {
      is.na(as.numeric(x))
    }))) {
      output$prediction_result <- renderText({
        "No prediction available"
      })
    } else {
      mean_age <- 0.08022222222222224
      std_dev_age <- 0.7018389534677384
      mean_abdomen <- 0.10706620209059248
      std_dev_abdomen <- 0.7078971247324045
      mean_wrist <- -0.05466666666666733
      std_dev_wrist <- 0.7535875819314118
      mean_bodyfat <- 0.001639560439560384
      std_dev_bodyfat <- 0.6652342205005188
      
      scaled_age <- (as.numeric(input$age) - mean_age) / std_dev_age
      scaled_abdomen <- (as.numeric(input$abdomen) - mean_abdomen) / std_dev_abdomen
      scaled_wrist <- (as.numeric(input$wrist) - mean_wrist) / std_dev_wrist
      
      new_data <- data.frame(
        AGE = scaled_age,
        ABDOMEN = scaled_abdomen,
        WRIST = scaled_wrist
      )
      scaled_prediction <- predict(lm5.1, newdata = new_data)
      
      original_scale_prediction <- (scaled_prediction * std_dev_bodyfat) + mean_bodyfat
      
      output$prediction_result <- renderText({
        paste("Predicted BODYFAT:", round(original_scale_prediction, 2))
      })
      
      #prediction <- predict(lm5.1, newdata = new_data)
      
      #output$prediction_result <- renderText({
        #paste("Predicted BODYFAT:", round(prediction, 2))
      #})
    }
  })
}

shinyApp(ui, server)
