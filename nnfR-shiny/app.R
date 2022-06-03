library(shiny)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(kableExtra)
#devtools::install_github("hans-elliott99/nnfR") ##if needed
library(nnfR)
##Deployment:
#library(rsconnect)
#rsconnect::deployApp('nnfR-shiny')


# Define UI ----
ui = fluidPage(
  tags$style(type = "text/css", "body {padding-left: 10px;}"),
  tags$head(
    tags$style(HTML(
    "
    @import url('https://fonts.googleapis.com/css2?family=Nuosu+SIL&family=Quicksand:wght@300&family=Roboto+Mono:wght@100&display=swap');
    body {
    font-family: 'Nuosu SIL', serif;
    }
    "
  )),
  ),
  # App title ----
  titlePanel("Classification with an Artificial Neural Network", 
             windowTitle = "nnfR"),
  #Introduction text ----
  shiny::verticalLayout(
    mainPanel(
      h3("Train a neural network to learn the different classes
        in a simulated dataset."),
      h4("Change the challenge of the task by tweaking
        the size of the dataset and the number of classes (",
         span("Hint:", style="font-style:italic"),
         "try 2 classes for a binary classification setup)."),
      h4("Change the structure of the network by tweaking the number of neurons in
        the hidden layer and the dropout rate."),
      h4("Tune the network by experimenting with different optimizer 
        hyperparameters and training epochs."),
      br(),
      p("This app was built from a custom neural network built entirely from R!
        See the nnfR package for more information."),
      br(),
      tags$a(href="https://github.com/hans-elliott99/nnfR", target="_blank",
             "nnfR package",
             style = "color:red"),
      tags$a(href="https://hans-elliott99.github.io/",target="_blank",
             "by Hans Elliott")
    )
  ),
  # Sidebar layout for simulating data ----
  sidebarLayout(
    
    # Sidebar panel for inputs
    sidebarPanel(
      h4("The Task"),
      # Input: Slider for the number of data points per class
      sliderInput(inputId = "data_points",
                  label = "Data points per class",
                  min = 50,
                  max = 200,
                  value = 50),
      # Input: Slide for the number of classes in data
      sliderInput(inputId = "num_class",
                  label = "Number of Classes",
                  min = 2,
                  max = 10,
                  value = 3), 
      width = 4
    ),
    # Main panel for displaying plot outputs
    mainPanel(
      # Output: Spiral Data plot
      plotOutput(outputId = "spiral_plot"),
      width = 6
    )
  ),
  
  # Sidebar layout with inputs for model ----
  sidebarLayout(
    
    # Side bar panel for model inputs & params
    sidebarPanel(
    h4("Hidden Layer"),
    sliderInput(inputId = "num_neurons",
                label = "Number of Neurons",
                min = 1,
                max = 5,
                value = 3),
    sliderInput(inputId = "dropout",
                label = "Dropout rate",
                min = 0,
                max = 0.9,
                value = 0),
    width = 4
    ),
    # Main panel for displaying model
    mainPanel(
      h4("Network Structure"),
      shiny::uiOutput(outputId = "model"), width = 6
    )
  ),
  
  # Sidebar layout with inputs for optimizer ----
  sidebarLayout(
    sidebarPanel(
    h4("Optimizer: Adam"),
    tags$a(href = "https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/#:~:text=Adam%20Configuration%20Parameters,used%20with%20Adam.",
           target="_blank", "An introduction to Adam."),
    shiny::sliderInput(inputId = "lr",
                       label = "Learning rate",
                       min = 0,
                       max = 1,
                       value = 0.01,
                       step = 0.001
                       ),
    shiny::numericInput(inputId = "lr_decay",
                        label = "Learning rate decay",
                        min = 0,
                        max = 0.1,
                        value = 1e-3,
                        step = 1e-3),
    shiny::sliderInput(inputId = "beta1",
                        label = "Beta 1",
                        min = 0,
                        max = 0.999,
                        value = 0.2
                        ),
    shiny::sliderInput(inputId = "beta2",
                        label = "Beta 2",
                        min = 0,
                        max = 0.999,
                        value = 0.2),
    h4("Training"),
    sliderInput(inputId = "epochs",
                label = "Number of Training Epochs",
                min = 1,
                max = 100,
                value = 10,
                step = 1)
    
    ),
    mainPanel(
      shiny::plotOutput(outputId = "pred_plot"),width = 6
    )
  ),
  # Sidebar layout with training inputs ---- 
  verticalLayout(
    # sidebarPanel(
    #   h4("Training"),
    #   sliderInput(inputId = "epochs",
    #               label = "Number of Training Epochs",
    #               min = 1,
    #               max = 100,
    #               value = 10,
    #               step = 1)
    # ),
    # Main panel for displaying the training results
    mainPanel(h3("Training Performance"),
      fluidRow(
        splitLayout(cellWidths = c("50%", "50%"), 
                    plotOutput("loss_plot"), 
                    plotOutput("accuracy_plot"))
              ), width = 12
    ),
    fluidRow(
      h4("Confusion Matrix"),
      h6("rows = predicted class, columns = true class"),
      splitLayout(cellWidths = c("50%","50%"),
                  shiny::uiOutput(outputId = "conf_mat"),
                  shiny::uiOutput(outputId = "performance"),
                  )
    )
  )
)

# Server----
server = function(input, output) { 
  #Sim data ----
  sim_spiral = reactive({
    spiral_data = nnfR::sim_spiral_data(N = input$data_points, 
                                        K = input$num_class)
    if (input$num_class <= 2) spiral_data$label = spiral_data$label - 1
    return(spiral_data)
  })
  
  #Build model ----
  build_model = reactive({
    spiral_data = sim_spiral()
    
    if (input$num_class <= 2){ 
      loss_fn = "binary_crossentropy"
      activ_fn = "sigmoid"
      final_neurons = 1
    } else {
      loss_fn = "categorical_crossentropy"
      activ_fn = "softmax_crossentropy" 
      final_neurons = input$num_class
    }
    
    
    model = nnfR::initialize_model() %>%
      #layer1
      nnfR::add_dense(n_inputs = ncol(spiral_data[, c("x", "y")]),
                      n_neurons = input$num_neurons,
                      dropout_rate = input$dropout
                      ) %>%
      nnfR::add_activation("relu") %>%
      #output layer
      nnfR::add_dense(n_inputs = input$num_neurons,
                      n_neurons = final_neurons) %>%
      nnfR::add_activation(paste0(activ_fn)) %>%
      #loss function
      nnfR::add_loss(paste0(loss_fn))
  })
  
  #Plot data----
  output$spiral_plot = renderPlot({
    
    spiral_data = sim_spiral()
    
    spiral_data %>%
      ggplot(aes(x = x, y = y, color = as.factor(label))) +
      geom_point(alpha = 0.7) +
      ggthemes::scale_color_calc() +
      #hrbrthemes::scale_color_ft() +
      labs(title = "Simulated Classification Task") +
      #hrbrthemes::theme_ipsum() +
      ggthemes::theme_tufte() + 
      theme(
        legend.position = "none",
        plot.title = element_text(size = 12),
        panel.background = element_rect(fill = "gray98", color = "gray90"),
        panel.grid.major = element_line(size = 0.1, color = "gray90")
      )
  })
  
  #Print Model ----
  output$model = renderTable({
    
    output_neurons = ifelse(input$num_class <= 2, 1, input$num_class)
    
    
    model = build_model()
    model_info = tibble(
      "Layer" = c("Hidden Layer", "Output Layer"),
      "Number of Inputs" = c("2 (x & y coordinates)", input$num_neurons),
      "Number of Neurons" = c(input$num_neurons, output_neurons),
      "Dropout Rate" = c(input$dropout, 0),
      "Activation Function" = c("reLU", model$layer4$class),
      "Loss Function" = c(" -- ", model$layer5$class)
    )
    
    print(model_info)
  }, striped = TRUE)

  #Train model ----
  fit_model = reactive({

    ##Data
    spiral_data = sim_spiral()
    X = as.matrix(spiral_data[,c("x","y")])
    y = spiral_data$label
    
    ##Validation data
      validation = sim_spiral_data(N = floor(0.5*input$data_points),
                                   K = input$num_class)
      X_val = as.matrix(validation[, c("x", "y")])
      y_val = validation$label
      if (input$num_class <= 2) y_val = y_val - 1

    ##Build Model
    model = build_model()
    ##Train Model
    fit = nnfR::train_model(model = model,
                            inputs = X,
                            y_true = y,
                            epochs = input$epochs,
                            optimizer = "adam",
                            learning_rate = input$lr,
                            lr_decay = input$lr_decay,
                            beta_1 = input$beta1, 
                            beta_2 = input$beta2,
                            metric_list = c("accuracy"),
                            validation_X = X_val,
                            validation_y = y_val,
                            epoch_print = input$epochs)
    return(fit)
  })
  
  #Plot predictions ----
  output$pred_plot = renderPlot({
    fit = fit_model()
    spiral_data = sim_spiral()
    spiral_data = spiral_data[,c("x","y")]
    data = cbind(spiral_data, pred = fit$predictions)  
    
    data %>%
      ggplot(aes(x = x, y = y, color = as.factor(pred))) +
      geom_point(alpha = 0.7) +
      ggthemes::scale_color_calc() +
      #hrbrthemes::scale_color_ft() +
      labs(title = "Predicted Classes") +
      #hrbrthemes::theme_ipsum() +
      ggthemes::theme_tufte() + 
      theme(
        legend.position = "none",
        plot.title = element_text(size = 15),
        panel.background = element_rect(fill = "gray98", color = "gray90"),
        panel.grid.major = element_line(size = 0.1, color = "gray90")
      )
  })
  
  #Plot loss ----
  output$loss_plot = renderPlot({
    fit = fit_model()
    
    label1 = paste("Loss:",
                   round(fit$final_metrics$loss, 3))
    label2 = paste("Validation Loss:", 
                   round(fit$final_metrics$validation_loss, 3))
    max_loss = max(fit$metrics$loss)
    sec_loss = sort(fit$metrics$loss)[length(fit$metrics$loss) - 2]
    
    fit$metrics %>%
      ggplot(aes(x = epoch)) +
      geom_line(aes(y = loss), color = "blue") +
      geom_line(aes(y = validation_loss), color = "red") +
      annotate("text", x = input$epochs-2, y = max_loss, 
               label = label1, color = "blue", alpha = 0.5) +
      annotate("text", x = input$epochs-2, y = sec_loss,
               label = label2, color = "red", alpha = 0.5) +
      #hrbrthemes::theme_ipsum() +
      labs(title = "Loss", y = "Loss") +
      ggthemes::theme_tufte() + 
      theme(
        legend.position = "none",
        panel.grid.major = element_line(size = 0.1, color = "gray90")
      )
    
  })
  
  #Plot accuracy ----
  output$accuracy_plot = renderPlot({
    fit = fit_model()
    label1 = paste("Acc:",
                   round(fit$final_metrics$accuracy, 3))
    label2 = paste("Validation Acc:", 
                   round(fit$validation_metrics$accuracy, 3))
    first_acc = sort(fit$metrics$accuracy)[1]
    third_acc = sort(fit$metrics$accuracy)[3]
    
    fit$metrics %>%
    ggplot(aes(x = epoch, y = accuracy)) +
      geom_line(color = "blue") +
      annotate("text", x = input$epochs-2, y = first_acc, 
               label = label1, color = "blue", alpha = 0.5) +
      annotate("text", x = input$epochs-2, y = third_acc,
               label = label2, color = "red", alpha = 0.5) +
      #hrbrthemes::theme_ipsum() +
      labs(title = "Accuracy", y = "Training Accuracy") +
      ggthemes::theme_tufte() + 
      theme(
        legend.position = "none",
        panel.grid.major = element_line(size = 0.1, color = "gray90")
      )
    
    })
  
  #Calculate performance metrics ----
  metrics = reactive({
    spiral_data  = sim_spiral()
    fit = fit_model()

    perf = classess(truths = spiral_data$label,
                    predictions = fit$predictions)
    return(perf)
  })
  
  #Print confusion matrix ----
  output$conf_mat = renderTable({
    perf = metrics()
    #format confusion matrix
    conf_mat = matrix(perf$conf_mat, 
                      ncol = ncol(perf$conf_mat), nrow = nrow(perf$conf_mat), 
                     dimnames = dimnames(perf$conf_mat))
    print(conf_mat)
    
  }, rownames = TRUE, colnames = TRUE, bordered = TRUE)
  

  #Print performance metrics ----
  output$performance = renderTable({
    perf = metrics()
    print(perf$metrics)
  }, striped = T, bordered = T)
  
}


shinyApp(ui = ui, server = server)



