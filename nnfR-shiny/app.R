library(shiny)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(kableExtra)
#devtools::install_github("hans-elliott99/nnfR") ##if need to update
library(nnfR)
##Deployment:
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
             windowTitle = "nnfR-shiny"),
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
      p("This app was built on a homemade neural network programmed entirely in R!
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
      br(),
      p("Each point in the plot represents one observation. Every observation 
        belongs to one class out of the chosen number. The neural network will
        be fed this data and be trained to learn which class each observation 
        belongs to."),
      p("Increasing the amount of data might improve the model's ability to 
        learn the classes, but is also more computationally demanding and could
        slow down training times."),
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
                max = 64,
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
      shiny::uiOutput(outputId = "model"), 
      p("We control the number of neurons in the hidden layer. Increasing the number
        makes for a more complex neural network, which could improve performance (but again,
        is more computationally demanding). The number of neurons in the output layer
        is equal to the number of classes we want to predict (except for in binary
        clasification, where we'll use 1 output neuron)."),
      p("Dropout randomly disables some neurons during training, which might help
        activate more neurons and make the model more generalizable. We do
        not apply dropout to the output layer."),
      p("Activation functions are applied to the output of each neuron. ReLU is a common
        function used to introduce nonlinearity. Softmax and sigmoid activation functions
        coerce the outputs to be between 0 and 1, so that the model predicts the probability 
        an observation belongs to each class. It's final prediction is whichever class receives the 
        highest probability."),
      p("Loss functions are special measures of error. We compare the model's predicted
        probabilities to the true classes in order to determine how the model's 
        parameters should be altered before the next epoch."),
      width = 6
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
                       value = 0.1,
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
                step = 1),
    p("An epoch is just one complete pass of the simulated data through the neural network.
      More epochs give the model more chances to learn, but could lead to overfitting.")
    ),
    mainPanel(
      shiny::plotOutput(outputId = "pred_plot"),
      br(),
      br(),
      br(),
      p("Colored regions attempt to depict the model's decision boundaries (i.e.,
      what your model would predict for a point within in that region). 
      Regions containing points of a different class suggest areas of uncertainty."),
      width = 6,
    )
  ),
  # Sidebar layout with training inputs ---- 
  verticalLayout(
    # Main panel for displaying the training results
    mainPanel(h3("Training Performance"),
              p("Here we can see the model's performance over the course of training.
                We hope to see loss (a measure of error) decreasing over time while 
                accuracy should be increasing. Validation data gives us a sense of how the model would do if it was
                used to predict the classes of observations it had never seen before."),
              p("Other metrics give insight into the model's ability to predict each class."),
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
                  shiny::uiOutput(outputId = "performance")
                  )
    )
  )
)

# Server----
server = function(input, output) { 
  #Sim data ----
  sim_spiral = reactive({
    set.seed(159)
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
    if (model$layer4$class == "softmax_crossentropy") a = "softmax"
    if (model$layer4$class == "sigmoid") a = "sigmoid"
    
    model_info = tibble(
      "Layer" = c("Hidden Layer", "Output Layer"),
      "Number of Inputs" = c("2 (x & y coordinates)", input$num_neurons),
      "Number of Neurons" = c(input$num_neurons, output_neurons),
      "Dropout Rate" = c(input$dropout, 0),
      "Activation Function" = c("reLU", a),
      "Loss Function" = c(" -- ", model$layer5$class)
    )
    
    print(model_info)
  }, striped = T, bordered = T)

  #Train model & make predictions ----
  fit_model = reactive({
    set.seed(159)
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
    
    predict = nnfR::test_model(model = model,
                               trained_model = fit,
                               X_test = X,
                               y_test = y,
                               metric_list = c("accuracy"))
    fit = list("fit"=fit, "predict"=predict)
    #return(fit)
  })
  
  #Plot predictions ----
  output$pred_plot = renderPlot({
    fit_model = fit_model()
    fit = fit_model$fit
    model = build_model()
    spiral_data = sim_spiral()
    spiral_data = spiral_data[,c("x","y")]
    data = cbind(spiral_data, pred = fit_model$predict$predictions)  
    
    #x = cbind(spiral_data, pred = fit_categ$predictions)
    #make decision boundary area
    ##make grid of x and y coords
    x_vals = seq(min(spiral_data$x), max(spiral_data$x), by = 0.05)
    y_vals = seq(min(spiral_data$y), max(spiral_data$y), by = 0.05)
    full_grid = as.matrix(expand.grid(x_vals, y_vals))
    ##predict onto grid
    pred = test_model(model = model,
                      trained_model = fit,
                      X_test = full_grid)
    ##create df for ggplot
    full_grid = data.frame(full_grid, pred = pred$predictions)
    
    data %>%
      ggplot(aes(x = x, y = y, color = as.factor(pred))) +
      ##plot decision boundaries -density shows overlap...
        # stat_density_2d(data = full_grid,
        #           aes(x = Var1, y = Var2, fill = as.factor(pred)),
        #           alpha = 0.08, geom = "polygon", linetype = 0) +
      geom_raster(data = full_grid,
                  aes(x = Var1, y = Var2, fill = as.factor(pred)),
                  alpha = 0.2) +
      geom_point(alpha = 1) +
      ggthemes::scale_color_calc() +
      ggthemes::scale_fill_calc() + 
      #hrbrthemes::scale_color_ft() +
      labs(title = "Predicted Classes", color = "", fill = "") +
      #hrbrthemes::theme_ipsum() +
      ggthemes::theme_tufte() + 
      theme(
        legend.position = "bottom",
        plot.title = element_text(size = 15),
        panel.background = element_rect(fill = "gray98", color = "gray90"),
        panel.grid.major = element_line(size = 0.1, color = "gray90")
      )
  },height = 450)
  
  #Plot loss ----
  output$loss_plot = renderPlot({
    fit = fit_model()
    fit = fit$fit
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
    fit = fit$fit
    label1 = paste("Acc:",
                   round(fit$final_metrics$accuracy, 3))
    label2 = paste("Validation Acc:", 
                   round(fit$final_metrics$val_accuracy, 3))
    first_acc = sort(fit$metrics$accuracy)[1]
    third_acc = sort(fit$metrics$accuracy)[3]
    
    fit$metrics %>%
    ggplot() +
      geom_line(aes(x = epoch, y = accuracy), color = "blue") +
      geom_line(aes(x = epoch, y = val_accuracy), color = "red") +
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
    
    perform = classess(truths = spiral_data$label,
                    predictions = fit$predict$predictions)
    return(perform)
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



