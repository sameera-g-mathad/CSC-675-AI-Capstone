import { KeyboardAvoidingView, Platform, SafeAreaView, StyleSheet, Text, View } from "react-native";
import React, { useContext, useEffect } from "react";
import { useNavigation, useLocalSearchParams, useRouter } from "expo-router";
import Chatbox from "@/app/_components/Chatbox";
import ChatHistContext from "@/app/_context/ChathistContext";

export default function Chat() {
    const { messages, askQuery } = useContext(ChatHistContext)
    const navigation = useNavigation();
    const { param, title } = useLocalSearchParams();
    useEffect(() => {
        navigation.setOptions({
            header: () =>
            (
                <View style={{
                    height: 90,
                    padding: 10,
                    justifyContent: 'center'
                }}>
                    < Text >{title}</ Text>
                </View>
            )
        })
    }, [])
    return (
        <SafeAreaView style={styles.viewContainer}>
            <KeyboardAvoidingView
                behavior={Platform.OS === "ios" ? "padding" : "height"}
                keyboardVerticalOffset={90}
                style={{ flex: 1 }}>
                <View style={styles.chatContainer}>
                    <Text>{param}</Text>
                </View>
                <View>
                    <Chatbox />
                </View>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    viewContainer: {
        flex: 1,
        backgroundColor: "blue"
    },
    chatContainer: {
        flex: 1,
    },

});
